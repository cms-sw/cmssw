//
// Author: Felice Pantaleo, CERN
//

#include <functional>

#include "CommonTools/Utils/interface/DynArray.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitPredictionFromCircle.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"

#include "CAHitQuadrupletGeneratorGPU.h"
#include "CAHitQuadrupletGeneratorKernels.h"

namespace {

  template <typename T> T sqr(T x) { return x * x; }

} // namespace

using namespace std;

constexpr unsigned int CAHitQuadrupletGeneratorGPU::minLayers;

CAHitQuadrupletGeneratorGPU::CAHitQuadrupletGeneratorGPU(
    const edm::ParameterSet &cfg,
    edm::ConsumesCollector &iC)
  : extraHitRPhitolerance(cfg.getParameter<double>("extraHitRPhitolerance")), // extra window in ThirdHitPredictionFromCircle range (divide by R to get phi)
    maxChi2(cfg.getParameter<edm::ParameterSet>("maxChi2")),
    fitFastCircle(cfg.getParameter<bool>("fitFastCircle")),
    fitFastCircleChi2Cut(cfg.getParameter<bool>("fitFastCircleChi2Cut")),
    useBendingCorrection(cfg.getParameter<bool>("useBendingCorrection")),
    caThetaCut(cfg.getParameter<double>("CAThetaCut")),
    caPhiCut(cfg.getParameter<double>("CAPhiCut")),
    caHardPtCut(cfg.getParameter<double>("CAHardPtCut"))
{
  edm::ParameterSet comparitorPSet = cfg.getParameter<edm::ParameterSet>("SeedComparitorPSet");
  std::string comparitorName = comparitorPSet.getParameter<std::string>("ComponentName");
  if (comparitorName != "none") {
    theComparitor.reset(SeedComparitorFactory::get()->create(comparitorName, comparitorPSet, iC));
  }
}

void CAHitQuadrupletGeneratorGPU::fillDescriptions(edm::ParameterSetDescription &desc) {
  desc.add<double>("extraHitRPhitolerance", 0.1);
  desc.add<bool>("fitFastCircle", false);
  desc.add<bool>("fitFastCircleChi2Cut", false);
  desc.add<bool>("useBendingCorrection", false);
  desc.add<double>("CAThetaCut", 0.00125);
  desc.add<double>("CAPhiCut", 10);
  desc.add<double>("CAHardPtCut", 0);
  desc.addOptional<bool>("CAOnlyOneLastHitPerLayerFilter")->setComment(
      "Deprecated and has no effect. To be fully removed later when the "
      "parameter is no longer used in HLT configurations.");
  edm::ParameterSetDescription descMaxChi2;
  descMaxChi2.add<double>("pt1", 0.2);
  descMaxChi2.add<double>("pt2", 1.5);
  descMaxChi2.add<double>("value1", 500);
  descMaxChi2.add<double>("value2", 50);
  descMaxChi2.add<bool>("enabled", true);
  desc.add<edm::ParameterSetDescription>("maxChi2", descMaxChi2);

  edm::ParameterSetDescription descComparitor;
  descComparitor.add<std::string>("ComponentName", "none");
  descComparitor.setAllowAnything();    // until we have moved SeedComparitor to EDProducers too
  desc.add<edm::ParameterSetDescription>("SeedComparitorPSet", descComparitor);
}

void CAHitQuadrupletGeneratorGPU::initEvent(edm::Event const& ev, edm::EventSetup const& es) {
  if (theComparitor)
    theComparitor->init(ev, es);
  bField_ = 1 / PixelRecoUtilities::fieldInInvGev(es);
}


CAHitQuadrupletGeneratorGPU::~CAHitQuadrupletGeneratorGPU() {
    deallocateOnGPU();
}

void CAHitQuadrupletGeneratorGPU::hitNtuplets(
    TrackingRegion const& region,
    HitsOnCPU const& hh,
    edm::EventSetup const& es,
    bool doRiemannFit,
    bool transferToCPU,
    cudaStream_t cudaStream)
{
  hitsOnCPU = &hh;
  int index = 0;
  launchKernels(region, index, hh, doRiemannFit, transferToCPU, cudaStream);
}

void CAHitQuadrupletGeneratorGPU::fillResults(
    const TrackingRegion &region, SiPixelRecHitCollectionNew const & rechits,
    std::vector<OrderedHitSeeds> &result, const edm::EventSetup &es)
{
  hitmap_.clear();
  auto const & rcs = rechits.data();
  for (auto const & h : rcs) hitmap_.add(h, &h);

  assert(hitsOnCPU);
  auto nhits = hitsOnCPU->nHits;
  int index = 0;

  auto const & foundQuads = fetchKernelResult(index);
  unsigned int numberOfFoundQuadruplets = foundQuads.size();
  const QuantityDependsPtEval maxChi2Eval = maxChi2.evaluator(es);

  // re-used throughout
  std::array<float, 4> bc_r;
  std::array<float, 4> bc_z;
  std::array<float, 4> bc_errZ2;
  std::array<GlobalPoint, 4> gps;
  std::array<GlobalError, 4> ges;
  std::array<bool, 4> barrels;
  std::array<BaseTrackerRecHit const*, 4> phits;

  // loop over quadruplets
  for (unsigned int quadId = 0; quadId < numberOfFoundQuadruplets; ++quadId) {
    auto isBarrel = [](const unsigned id) -> bool {
      return id == PixelSubdetector::PixelBarrel;
    };
    bool bad = false;
    for (unsigned int i = 0; i < 4; ++i) {
      auto k = foundQuads[quadId][i];
      assert(k<int(nhits));
      auto hp = hitmap_.get((*hitsOnCPU).detInd[k],(*hitsOnCPU).mr[k], (*hitsOnCPU).mc[k]);
      if (hp==nullptr) {
        bad=true;
        break;
      }
      phits[i] = static_cast<BaseTrackerRecHit const *>(hp);
      auto const &ahit = *phits[i];
      gps[i] = ahit.globalPosition();
      ges[i] = ahit.globalPositionError();
      barrels[i] = isBarrel(ahit.geographicalId().subdetId());

    }
    if (bad) continue;

    // TODO:
    // - if we decide to always do the circle fit for 4 hits, we don't
    //   need ThirdHitPredictionFromCircle for the curvature; then we
    //   could remove extraHitRPhitolerance configuration parameter
    ThirdHitPredictionFromCircle predictionRPhi(gps[0], gps[2],
        extraHitRPhitolerance);
    const float curvature = predictionRPhi.curvature(
        ThirdHitPredictionFromCircle::Vector2D(gps[1].x(), gps[1].y()));
    const float abscurv = std::abs(curvature);
    const float thisMaxChi2 = maxChi2Eval.value(abscurv);
    if (theComparitor) {
      SeedingHitSet tmpTriplet(phits[0],  phits[1],  phits[3]);
      if (!theComparitor->compatible(tmpTriplet)) {
        continue;
      }
    }

    float chi2 = std::numeric_limits<float>::quiet_NaN();
    // TODO: Do we have any use case to not use bending correction?
    if (useBendingCorrection) {
      // Following PixelFitterByConformalMappingAndLine
      const float simpleCot = (gps.back().z() - gps.front().z()) /
        (gps.back().perp() - gps.front().perp());
      const float pt = 1.f / PixelRecoUtilities::inversePt(abscurv, es);
      for (int i = 0; i < 4; ++i) {
        const GlobalPoint &point = gps[i];
        const GlobalError &error = ges[i];
        bc_r[i] = sqrt(sqr(point.x() - region.origin().x()) +
            sqr(point.y() - region.origin().y()));
        bc_r[i] += pixelrecoutilities::LongitudinalBendingCorrection(pt, es)(
            bc_r[i]);
        bc_z[i] = point.z() - region.origin().z();
        bc_errZ2[i] =
          (barrels[i]) ? error.czz() : error.rerr(point) * sqr(simpleCot);
      }
      RZLine rzLine(bc_r, bc_z, bc_errZ2, RZLine::ErrZ2_tag());
      chi2 = rzLine.chi2();
    } else {
      RZLine rzLine(gps, ges, barrels);
      chi2 = rzLine.chi2();
    }
    if (edm::isNotFinite(chi2) || chi2 > thisMaxChi2) {
      continue;
    }
    // TODO: Do we have any use case to not use circle fit? Maybe
    // HLT where low-pT inefficiency is not a problem?
    if (fitFastCircle) {
      FastCircleFit c(gps, ges);
      chi2 += c.chi2();
      if (edm::isNotFinite(chi2))
        continue;
      if (fitFastCircleChi2Cut && chi2 > thisMaxChi2)
        continue;
    }
    result[index].emplace_back(phits[0],  phits[1],  phits[2],  phits[3]);

  } // end loop over quads
}

void CAHitQuadrupletGeneratorGPU::launchKernels(const TrackingRegion &region,
                                                int regionIndex, HitsOnCPU const & hh,
                                                bool doRiemannFit,
                                                bool transferToCPU,
                                                cudaStream_t cudaStream)
{
  assert(regionIndex < maxNumberOfRegions_);
  assert(0==regionIndex);

  h_foundNtupletsVec_[regionIndex]->reset();

  auto nhits = hh.nHits;
  assert(nhits <= PixelGPUConstants::maxNumberOfHits);
  auto blockSize = 64;
  auto numberOfBlocks = (maxNumberOfDoublets_ + blockSize - 1)/blockSize;
  //kernel_connect<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
  wrapperConnect(numberOfBlocks, blockSize, cudaStream,
      d_foundNtupletsVec_[regionIndex], // needed only to be reset, ready for next kernel
      hh.gpu_d,
      device_theCells_, device_nCells_,
      device_isOuterHitOfCell_,
      region.ptMin(),
      region.originRBound(), caThetaCut, caPhiCut, caHardPtCut,
      maxNumberOfDoublets_, PixelGPUConstants::maxNumberOfHits);

  //kernel_find_ntuplets<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
  wrapperFindNtuplets(numberOfBlocks, blockSize, cudaStream,
      device_theCells_, device_nCells_,
      d_foundNtupletsVec_[regionIndex],
      4, maxNumberOfDoublets_);

  numberOfBlocks = (std::max(int(nhits), maxNumberOfDoublets_) + blockSize - 1)/blockSize;
  //kernel_checkOverflows<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
  wrapperCheckOverflows(numberOfBlocks, blockSize, cudaStream,
      d_foundNtupletsVec_[regionIndex],
      device_theCells_, device_nCells_,
      device_isOuterHitOfCell_, nhits,
      maxNumberOfDoublets_);

  // kernel_print_found_ntuplets<<<1, 1, 0, cudaStream>>>(d_foundNtupletsVec_[regionIndex], 10);
  // wrapperPrintFoundNtuplets(cudaStream, d_foundNtupletsVec_[regionIndex], 10);

  if (doRiemannFit) {
    //kernelFastFitAllHits<<<numberOfBlocks, 512, 0, cudaStream>>>(
    wrapperFastFitAllHits(numberOfBlocks, 512, cudaStream,
        d_foundNtupletsVec_[regionIndex], hh.gpu_d, 4, bField_, helix_fit_resultsGPU_,
        hitsGPU_, hits_covGPU_, circle_fit_resultsGPU_, fast_fit_resultsGPU_,
        line_fit_resultsGPU_);

    blockSize = 256;
    numberOfBlocks = (maxNumberOfQuadruplets_ + blockSize - 1) / blockSize;

    //kernelCircleFitAllHits<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
    wrapperCircleFitAllHits(numberOfBlocks, blockSize, cudaStream,
        d_foundNtupletsVec_[regionIndex], 4, bField_, helix_fit_resultsGPU_,
        hitsGPU_, hits_covGPU_, circle_fit_resultsGPU_, fast_fit_resultsGPU_,
        line_fit_resultsGPU_);

    //kernelLineFitAllHits<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
    wrapperLineFitAllHits(numberOfBlocks, blockSize, cudaStream,
        d_foundNtupletsVec_[regionIndex], bField_, helix_fit_resultsGPU_,
        hitsGPU_, hits_covGPU_, circle_fit_resultsGPU_, fast_fit_resultsGPU_,
        line_fit_resultsGPU_);
  }

  if (transferToCPU) {
    cudaCheck(cudaMemcpyAsync(h_foundNtupletsVec_[regionIndex], d_foundNtupletsVec_[regionIndex],
                              sizeof(GPU::SimpleVector<Quadruplet>),
                              cudaMemcpyDeviceToHost, cudaStream));

    cudaCheck(cudaMemcpyAsync(h_foundNtupletsData_[regionIndex], d_foundNtupletsData_[regionIndex],
                              maxNumberOfQuadruplets_*sizeof(Quadruplet),
                              cudaMemcpyDeviceToHost, cudaStream));
  }
}

void CAHitQuadrupletGeneratorGPU::allocateOnGPU()
{
  //////////////////////////////////////////////////////////
  // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
  //////////////////////////////////////////////////////////

  cudaCheck(cudaMalloc(&device_theCells_,
             maxNumberOfLayerPairs_ * maxNumberOfDoublets_ * sizeof(GPUCACell)));
  cudaCheck(cudaMalloc(&device_nCells_, sizeof(uint32_t)));
  cudaCheck(cudaMemset(device_nCells_, 0, sizeof(uint32_t)));

  cudaCheck(cudaMalloc(&device_isOuterHitOfCell_,
             PixelGPUConstants::maxNumberOfHits * sizeof(GPU::VecArray<unsigned int, maxCellsPerHit_>)));
  cudaCheck(cudaMemset(device_isOuterHitOfCell_, 0,
             PixelGPUConstants::maxNumberOfHits * sizeof(GPU::VecArray<unsigned int, maxCellsPerHit_>)));

  h_foundNtupletsVec_.resize(maxNumberOfRegions_);
  h_foundNtupletsData_.resize(maxNumberOfRegions_);
  d_foundNtupletsVec_.resize(maxNumberOfRegions_);
  d_foundNtupletsData_.resize(maxNumberOfRegions_);

  // FIXME this could be rewritten with a single pair of cudaMallocHost / cudaMalloc
  for (int i = 0; i < maxNumberOfRegions_; ++i) {
    cudaCheck(cudaMallocHost(&h_foundNtupletsData_[i],  sizeof(Quadruplet) * maxNumberOfQuadruplets_));
    cudaCheck(cudaMallocHost(&h_foundNtupletsVec_[i],   sizeof(GPU::SimpleVector<Quadruplet>)));
    GPU::make_SimpleVector(h_foundNtupletsVec_[i], maxNumberOfQuadruplets_, h_foundNtupletsData_[i]);
    cudaCheck(cudaMalloc(&d_foundNtupletsData_[i],      sizeof(Quadruplet) * maxNumberOfQuadruplets_));
    cudaCheck(cudaMemset(d_foundNtupletsData_[i], 0x00, sizeof(Quadruplet) * maxNumberOfQuadruplets_));
    cudaCheck(cudaMalloc(&d_foundNtupletsVec_[i],       sizeof(GPU::SimpleVector<Quadruplet>)));
    auto tmp_foundNtuplets = GPU::make_SimpleVector<Quadruplet>(maxNumberOfQuadruplets_, d_foundNtupletsData_[i]);
    cudaCheck(cudaMemcpy(d_foundNtupletsVec_[i], & tmp_foundNtuplets, sizeof(GPU::SimpleVector<Quadruplet>), cudaMemcpyDefault));
  }

  // Riemann-Fit related allocations
  cudaCheck(cudaMalloc(&hitsGPU_, 48 * maxNumberOfQuadruplets_ * sizeof(Rfit::Matrix3xNd(3, 4))));
  cudaCheck(cudaMemset(hitsGPU_, 0x00, 48 * maxNumberOfQuadruplets_ * sizeof(Rfit::Matrix3xNd(3, 4))));

  cudaCheck(cudaMalloc(&hits_covGPU_, 48 * maxNumberOfQuadruplets_ * sizeof(Rfit::Matrix3Nd(12, 12))));
  cudaCheck(cudaMemset(hits_covGPU_, 0x00, 48 * maxNumberOfQuadruplets_ * sizeof(Rfit::Matrix3Nd(12, 12))));

  cudaCheck(cudaMalloc(&fast_fit_resultsGPU_, 48 * maxNumberOfQuadruplets_ * sizeof(Eigen::Vector4d)));
  cudaCheck(cudaMemset(fast_fit_resultsGPU_, 0x00, 48 * maxNumberOfQuadruplets_ * sizeof(Eigen::Vector4d)));

  cudaCheck(cudaMalloc(&circle_fit_resultsGPU_, 48 * maxNumberOfQuadruplets_ * sizeof(Rfit::circle_fit)));
  cudaCheck(cudaMemset(circle_fit_resultsGPU_, 0x00, 48 * maxNumberOfQuadruplets_ * sizeof(Rfit::circle_fit)));

  cudaCheck(cudaMalloc(&line_fit_resultsGPU_, maxNumberOfQuadruplets_ * sizeof(Rfit::line_fit)));
  cudaCheck(cudaMemset(line_fit_resultsGPU_, 0x00, maxNumberOfQuadruplets_ * sizeof(Rfit::line_fit)));

  cudaCheck(cudaMalloc(&helix_fit_resultsGPU_, sizeof(Rfit::helix_fit)*maxNumberOfQuadruplets_));
  cudaCheck(cudaMemset(helix_fit_resultsGPU_, 0x00, sizeof(Rfit::helix_fit)*maxNumberOfQuadruplets_));
}

void CAHitQuadrupletGeneratorGPU::deallocateOnGPU()
{
  for (size_t i = 0; i < h_foundNtupletsVec_.size(); ++i)
  {
    cudaFreeHost(h_foundNtupletsVec_[i]);
    cudaFreeHost(h_foundNtupletsData_[i]);
    cudaFree(d_foundNtupletsVec_[i]);
    cudaFree(d_foundNtupletsData_[i]);
  }

  cudaFree(device_theCells_);
  cudaFree(device_isOuterHitOfCell_);
  cudaFree(device_nCells_);

  // Free Riemann Fit stuff
  cudaFree(hitsGPU_);
  cudaFree(hits_covGPU_);
  cudaFree(fast_fit_resultsGPU_);
  cudaFree(circle_fit_resultsGPU_);
  cudaFree(line_fit_resultsGPU_);
  cudaFree(helix_fit_resultsGPU_);
}

void CAHitQuadrupletGeneratorGPU::cleanup(cudaStream_t cudaStream) {
  // this lazily resets temporary memory for the next event, and is not needed for reading the output
  cudaCheck(cudaMemsetAsync(device_isOuterHitOfCell_, 0,
                            PixelGPUConstants::maxNumberOfHits * sizeof(GPU::VecArray<unsigned int, maxCellsPerHit_>),
                            cudaStream));
  cudaCheck(cudaMemsetAsync(device_nCells_, 0, sizeof(uint32_t), cudaStream));
}

std::vector<std::array<int, 4>>
CAHitQuadrupletGeneratorGPU::fetchKernelResult(int regionIndex)
{
  assert(0==regionIndex);
  h_foundNtupletsVec_[regionIndex]->set_data(h_foundNtupletsData_[regionIndex]);

  std::vector<std::array<int, 4>> quadsInterface(h_foundNtupletsVec_[regionIndex]->size());
  for (int i = 0; i < h_foundNtupletsVec_[regionIndex]->size(); ++i) {
    for (int j = 0; j<4; ++j) quadsInterface[i][j] = (*h_foundNtupletsVec_[regionIndex])[i].hitId[j];
  }
  return quadsInterface;
}

void CAHitQuadrupletGeneratorGPU::buildDoublets(siPixelRecHitsHeterogeneousProduct::HitsOnCPU const & hh, cudaStream_t cudaStream)
{
  int threadsPerBlock = 64;     // FIXME gpuPixelDoublets::getDoubletsFromHistoMaxBlockSize
  int blocks = (3 * hh.nHits + threadsPerBlock - 1) / threadsPerBlock;
  wrapperDoubletsFromHisto(blocks, threadsPerBlock, cudaStream, device_theCells_, device_nCells_, hh.gpu_d, device_isOuterHitOfCell_);
}
