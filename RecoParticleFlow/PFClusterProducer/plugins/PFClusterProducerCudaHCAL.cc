#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

// Uncomment to enable GPU debugging
//#define DEBUG_GPU_HCAL

// Uncomment to enable LEGACY inputs
//#define DEBUG_WithLEGACY

#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TTree.h>

#include "CUDADataFormats/PFRecHitSoA/interface/PFRecHitCollection.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterBuilderBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterEnergyCorrectorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/RecHitTopologicalCleanerBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/SeedFinderBase.h"

#include "CudaPFCommon.h"
#include "DeclsForKernels.h"
#include "PFClusterCudaHCAL.h"

class PFClusterProducerCudaHCAL : public edm::stream::EDProducer<edm::ExternalWork> {
  typedef RecHitTopologicalCleanerBase RHCB;
  typedef InitialClusteringStepBase ICSB;
  typedef PFClusterBuilderBase PFCBB;
  typedef PFCPositionCalculatorBase PosCalc;

public:
  PFClusterProducerCudaHCAL(const edm::ParameterSet &);
  ~PFClusterProducerCudaHCAL() override;
  static void fillDescriptions(edm::ConfigurationDescriptions &);

  // the actual algorithm
  std::vector<std::unique_ptr<RecHitTopologicalCleanerBase>> _cleaners;
  std::unique_ptr<SeedFinderBase> _seedFinder;
  std::unique_ptr<InitialClusteringStepBase> _initialClustering;
  std::unique_ptr<PFClusterBuilderBase> _pfClusterBuilder;
  std::unique_ptr<PFCPositionCalculatorBase> _positionReCalc;
  std::unique_ptr<PFCPositionCalculatorBase> _allCellsPosCalc;
  std::unique_ptr<PFCPositionCalculatorBase> _positionCalc;
  std::unique_ptr<PFCPositionCalculatorBase> _allCellsPositionCalc;
  std::unique_ptr<PFClusterEnergyCorrectorBase> _energyCorrector;

  reco::PFClusterCollection __initialClusters;
  reco::PFClusterCollection __pfClusters;
  reco::PFClusterCollection __pfClustersFromCuda;
  reco::PFRecHitCollection __rechits;

  std::vector<int> __edgeId;
  std::vector<int> __edgeList;
  std::vector<int> __rh_isSeed;
  std::vector<int> __rh_isSeedCPU;  // CPU seeds
  std::vector<float> __rh_x;
  std::vector<float> __rh_y;
  std::vector<float> __rh_z;
  std::vector<float> __rh_eta;
  std::vector<float> __rh_phi;
  // rechit neighbours4, neighbours8 vectors
  std::vector<std::vector<int>> __rh_neighbours4;
  std::vector<std::vector<int>> __rh_neighbours8;

  std::vector<int> __pfrh_detIdGPU;
  std::vector<int> __pfrh_neighboursGPU;
  std::vector<int> __pfrh_neighboursDetIdGPU;

  std::vector<int> __pfcIter;
  std::vector<int> __nRHTopo;
  std::vector<int> __nSeedsTopo;
  std::vector<int> __nFracsTopo;

  bool initCuda = true;

  std::array<float, 9> GPU_timers;

  Int_t nRHperPFCTotal_CPU = 0;
  Int_t nRHperPFCTotal_GPU = 0;

private:
  void beginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &) override;
  void acquire(edm::Event const &, edm::EventSetup const &, edm::WaitingTaskWithArenaHolder) override;
  void produce(edm::Event &, const edm::EventSetup &) override;

  using IProductType = cms::cuda::Product<hcal::PFRecHitCollection<pf::common::DevStoragePolicy>>;
  edm::EDGetTokenT<IProductType> InputPFRecHitSoA_Token_;

  const bool _produceSoA;            // PFClusters in SoA format
  const bool _produceLegacy;         // PFClusters in legacy format

  // // options
  // const bool _prodInitClusters;

  edm::EDGetTokenT<reco::PFRecHitCollection> _rechitsLabel;

  cms::cuda::ContextState cudaState_;
  cudaStream_t cudaStream = cudaStreamDefault;

  PFClustering::HCAL::ConfigurationParameters cudaConfig_;
  PFClustering::common::CudaHCALConstants cudaConstants;
  PFClustering::HCAL::InputDataCPU inputCPU;
  PFClustering::HCAL::InputDataGPU inputGPU;

  PFClustering::HCAL::OutputDataCPU outputCPU;
  PFClustering::HCAL::OutputDataGPU outputGPU;

  PFClustering::HCAL::ScratchDataGPU scratchGPU;

  std::unique_ptr<reco::PFClusterCollection> pfClustersFromCuda;
  // CPU clusters
  std::unique_ptr<reco::PFClusterCollection> pfClusters;
};

#ifdef PFLOW_DEBUG
#define LOGVERB(x) edm::LogVerbatim(x)
#define LOGWARN(x) edm::LogWarning(x)
#define LOGERR(x) edm::LogError(x)
#define LOGDRESSED(x) edm::LogInfo(x)
#else
#define LOGVERB(x) LogTrace(x)
#define LOGWARN(x) edm::LogWarning(x)
#define LOGERR(x) edm::LogError(x)
#define LOGDRESSED(x) LogDebug(x)
#endif

PFClusterProducerCudaHCAL::PFClusterProducerCudaHCAL(const edm::ParameterSet& conf)
  : InputPFRecHitSoA_Token_{consumes<IProductType>(conf.getParameter<edm::InputTag>("PFRecHitsLabelIn"))},
    _produceSoA{conf.getParameter<bool>("produceSoA")},
    _produceLegacy{conf.getParameter<bool>("produceLegacy")},
    //KenH _prodInitClusters(conf.getUntrackedParameter<bool>("prodInitialClusters", false)),
    _rechitsLabel{consumes<reco::PFRecHitCollection>(conf.getParameter<edm::InputTag>("recHitsSource"))} {
  edm::ConsumesCollector cc = consumesCollector();

  //setup rechit cleaners
  const edm::VParameterSet& cleanerConfs = conf.getParameterSetVector("recHitCleaners");

  for (const auto& conf : cleanerConfs) {
    const std::string& cleanerName = conf.getParameter<std::string>("algoName");
    _cleaners.emplace_back(RecHitTopologicalCleanerFactory::get()->create(cleanerName, conf, cc));
  }

  // setup seed finding
  const edm::ParameterSet& sfConf = conf.getParameterSet("seedFinder");
  const std::string& sfName = sfConf.getParameter<std::string>("algoName");
  _seedFinder = SeedFinderFactory::get()->create(sfName, sfConf);

  const edm::VParameterSet& seedFinderConfs = sfConf.getParameterSetVector("thresholdsByDetector");

  cudaConstants.minFracInCalc = 0.0;
  cudaConstants.minAllowedNormalization = 0.0;

  //setup topo cluster builder
  const edm::ParameterSet& initConf = conf.getParameterSet("initialClusteringStep");
  const std::string& initName = initConf.getParameter<std::string>("algoName");
  _initialClustering = InitialClusteringStepFactory::get()->create(initName, initConf, cc);
  //setup pf cluster builder if requested
  const edm::ParameterSet& pfcConf = conf.getParameterSet("pfClusterBuilder");
  if (!pfcConf.empty()) {
    const std::string& pfcName = pfcConf.getParameter<std::string>("algoName");
    _pfClusterBuilder = PFClusterBuilderFactory::get()->create(pfcName, pfcConf, cc);
    /*
    if (pfcConf.exists("allCellsPositionCalc")) {
    const edm::ParameterSet& acConf = pfcConf.getParameterSet("allCellsPositionCalc");
    const std::string& algoac = acConf.getParameter<std::string>("algoName");
    _allCellsPosCalcCuda = PFCPositionCalculatorFactory::get()->create(algoac, acConf);
    */

    if (pfcConf.exists("positionCalc")) {
      const edm::ParameterSet& acConf = pfcConf.getParameterSet("positionCalc");
      const std::string& algoac = acConf.getParameter<std::string>("algoName");
      _positionCalc = PFCPositionCalculatorFactory::get()->create(algoac, acConf, cc);
      cudaConstants.minFracInCalc = (float)acConf.getParameter<double>("minFractionInCalc");
      cudaConstants.minAllowedNormalization = (float)acConf.getParameter<double>("minAllowedNormalization");
    }

    if (pfcConf.exists("allCellsPositionCalc")) {
      const edm::ParameterSet& acConf = pfcConf.getParameterSet("allCellsPositionCalc");
      const std::string& algoac = acConf.getParameter<std::string>("algoName");
      _allCellsPositionCalc = PFCPositionCalculatorFactory::get()->create(algoac, acConf, cc);
    }
  }
  //setup (possible) recalcuation of positions
  const edm::ParameterSet& pConf = conf.getParameterSet("positionReCalc");
  if (!pConf.empty()) {
    const std::string& pName = pConf.getParameter<std::string>("algoName");
    _positionReCalc = PFCPositionCalculatorFactory::get()->create(pName, pConf, cc);
  }
  // see if new need to apply corrections, setup if there.
  const edm::ParameterSet& cConf = conf.getParameterSet("energyCorrector");
  if (!cConf.empty()) {
    const std::string& cName = cConf.getParameter<std::string>("algoName");
    _energyCorrector = PFClusterEnergyCorrectorFactory::get()->create(cName, cConf);
  }

  cudaConstants.showerSigma2 = (float)std::pow(pfcConf.getParameter<double>("showerSigma"), 2.);
  const auto& recHitEnergyNormConf = pfcConf.getParameterSetVector("recHitEnergyNorms");
  for (const auto& pset : recHitEnergyNormConf) {
    const std::string& det = pset.getParameter<std::string>("detector");
    if (det == std::string("HCAL_BARREL1")) {
      const auto& recHitENorms = pset.getParameter<std::vector<double>>("recHitEnergyNorm");
      std::copy(recHitENorms.begin(), recHitENorms.end(), cudaConstants.recHitEnergyNormInvEB_vec);
      for (auto& x : cudaConstants.recHitEnergyNormInvEB_vec)
        x = std::pow(x, -1);  // Invert these values
    } else if (det == std::string("HCAL_ENDCAP")) {
      const auto& recHitENorms = pset.getParameter<std::vector<double>>("recHitEnergyNorm");
      std::copy(recHitENorms.begin(), recHitENorms.end(), cudaConstants.recHitEnergyNormInvEE_vec);
      for (auto& x : cudaConstants.recHitEnergyNormInvEE_vec)
        x = std::pow(x, -1);  // Invert these values
    } else
      std::cout << "Unknown detector when parsing recHitEnergyNorm: " << det << std::endl;
  }
  //float recHitEnergyNormEB = 0.08;
  //float recHitEnergyNormEE = 0.3;
  //float minFracToKeep = 0.0000001;
  cudaConstants.minFracToKeep = (float)pfcConf.getParameter<double>("minFractionToKeep");
  cudaConstants.minFracTot = (float)pfcConf.getParameter<double>("minFracTot");

  // Max PFClustering iterations
  cudaConstants.maxIterations = pfcConf.getParameter<unsigned>("maxIterations");

  cudaConstants.excludeOtherSeeds = pfcConf.getParameter<bool>("excludeOtherSeeds");

  cudaConstants.stoppingTolerance = (float)pfcConf.getParameter<double>("stoppingTolerance");

  cudaConstants.seedPt2ThresholdEB = -1;
  cudaConstants.seedPt2ThresholdEE = -1;
  for (const auto& pset : seedFinderConfs) {
    const std::string& det = pset.getParameter<std::string>("detector");
    if (det == std::string("HCAL_BARREL1")) {
      const auto& thresholds = pset.getParameter<std::vector<double>>("seedingThreshold");
      std::copy(thresholds.begin(), thresholds.end(), cudaConstants.seedEThresholdEB_vec);
      cudaConstants.seedPt2ThresholdEB =
          (float)std::pow(pset.getParameter<std::vector<double>>("seedingThresholdPt")[0], 2.0);

    } else if (det == std::string("HCAL_ENDCAP")) {
      const auto& thresholds = pset.getParameter<std::vector<double>>("seedingThreshold");
      std::copy(thresholds.begin(), thresholds.end(), cudaConstants.seedEThresholdEE_vec);
      cudaConstants.seedPt2ThresholdEE =
          (float)std::pow(pset.getParameter<std::vector<double>>("seedingThresholdPt")[0], 2.0);
    } else
      std::cout << "Unknown detector when parsing seedFinder: " << det << std::endl;
  }

  const auto& topoThresholdConf = initConf.getParameterSetVector("thresholdsByDetector");
  for (const auto& pset : topoThresholdConf) {
    const std::string& det = pset.getParameter<std::string>("detector");
    if (det == std::string("HCAL_BARREL1")) {
      const auto& thresholds = pset.getParameter<std::vector<double>>("gatheringThreshold");
      std::copy(thresholds.begin(), thresholds.end(), cudaConstants.topoEThresholdEB_vec);
    } else if (det == std::string("HCAL_ENDCAP")) {
      const auto& thresholds = pset.getParameter<std::vector<double>>("gatheringThreshold");
      std::copy(thresholds.begin(), thresholds.end(), cudaConstants.topoEThresholdEE_vec);
    } else
      std::cout << "Unknown detector when parsing initClusteringStep: " << det << std::endl;
  }

  if (pfcConf.exists("timeResolutionCalcEndcap")) {
    const edm::ParameterSet& endcapTimeResConf = pfcConf.getParameterSet("timeResolutionCalcEndcap");
    cudaConstants.endcapTimeResConsts.corrTermLowE = (float)endcapTimeResConf.getParameter<double>("corrTermLowE");
    cudaConstants.endcapTimeResConsts.threshLowE = (float)endcapTimeResConf.getParameter<double>("threshLowE");
    cudaConstants.endcapTimeResConsts.noiseTerm = (float)endcapTimeResConf.getParameter<double>("noiseTerm");
    cudaConstants.endcapTimeResConsts.constantTermLowE2 =
        (float)std::pow(endcapTimeResConf.getParameter<double>("constantTermLowE"), 2.0);
    cudaConstants.endcapTimeResConsts.noiseTermLowE = (float)endcapTimeResConf.getParameter<double>("noiseTermLowE");
    cudaConstants.endcapTimeResConsts.threshHighE = (float)endcapTimeResConf.getParameter<double>("threshHighE");
    cudaConstants.endcapTimeResConsts.constantTerm2 =
        (float)std::pow(endcapTimeResConf.getParameter<double>("constantTerm"), 2.0);
    cudaConstants.endcapTimeResConsts.resHighE2 =
        (float)std::pow(cudaConstants.endcapTimeResConsts.noiseTerm / cudaConstants.endcapTimeResConsts.threshHighE,
                        2.0) +
        cudaConstants.endcapTimeResConsts.constantTerm2;
  }

  if (pfcConf.exists("timeResolutionCalcBarrel")) {
    const edm::ParameterSet& barrelTimeResConf = pfcConf.getParameterSet("timeResolutionCalcBarrel");
    cudaConstants.barrelTimeResConsts.corrTermLowE = (float)barrelTimeResConf.getParameter<double>("corrTermLowE");
    cudaConstants.barrelTimeResConsts.threshLowE = (float)barrelTimeResConf.getParameter<double>("threshLowE");
    cudaConstants.barrelTimeResConsts.noiseTerm = (float)barrelTimeResConf.getParameter<double>("noiseTerm");
    cudaConstants.barrelTimeResConsts.constantTermLowE2 =
        (float)std::pow(barrelTimeResConf.getParameter<double>("constantTermLowE"), 2.0);
    cudaConstants.barrelTimeResConsts.noiseTermLowE = (float)barrelTimeResConf.getParameter<double>("noiseTermLowE");
    cudaConstants.barrelTimeResConsts.threshHighE = (float)barrelTimeResConf.getParameter<double>("threshHighE");
    cudaConstants.barrelTimeResConsts.constantTerm2 =
        (float)std::pow(barrelTimeResConf.getParameter<double>("constantTerm"), 2.0);
    cudaConstants.barrelTimeResConsts.resHighE2 =
        (float)std::pow(cudaConstants.barrelTimeResConsts.noiseTerm / cudaConstants.barrelTimeResConsts.threshHighE,
                        2.0) +
        cudaConstants.barrelTimeResConsts.constantTerm2;
  }
  cudaConstants.nNeigh = sfConf.getParameter<int>("nNeighbours");

  pfClustersFromCuda = std::make_unique<reco::PFClusterCollection>();

  // if (_prodInitClusters) {
  //   produces<reco::PFClusterCollection>("initialClusters");
  // }
  produces<reco::PFClusterCollection>();
}

PFClusterProducerCudaHCAL::~PFClusterProducerCudaHCAL() {
}

void PFClusterProducerCudaHCAL::fillDescriptions(edm::ConfigurationDescriptions& cdesc) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("PFRecHitsLabelIn", edm::InputTag("hltParticleFlowRecHitHBHE"));
  desc.add<bool>("produceSoA", true);
  desc.add<bool>("produceLegacy", true);
  // Prevents the producer and navigator parameter sets from throwing an exception
  // TODO: Replace with a proper parameter set description: twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideConfigurationValidationAndHelp
  desc.setAllowAnything();

  cdesc.addWithDefaultLabel(desc);
}

void PFClusterProducerCudaHCAL::beginLuminosityBlock(const edm::LuminosityBlock& lumi, const edm::EventSetup& es) {
  /* KenH
  _initialClustering->update(es);
  if (_pfClusterBuilder)
    _pfClusterBuilder->update(es);
  if (_positionReCalc)
    _positionReCalc->update(es);
  */
  initCuda = true;  // (Re)initialize cuda arrays
}

void PFClusterProducerCudaHCAL::acquire(edm::Event const& event,
                                        edm::EventSetup const& setup,
                                        edm::WaitingTaskWithArenaHolder holder) {
  // Creates a new Cuda stream
  // TODO: Reuse stream from GPU PFRecHitProducer by passing input product as first arg
  // cmssdt.cern.ch/lxr/source/HeterogeneousCore/CUDACore/interface/ScopedContext.h#0101
  //cms::cuda::ScopedContextAcquire ctx{event.streamID(), std::move(holder), cudaState_};
  //cms::cuda::ScopedContextAcquire ctx{event.streamID(), std::move(holder)};
  auto const& PFRecHitsProduct = event.get(InputPFRecHitSoA_Token_);
  cms::cuda::ScopedContextAcquire ctx{PFRecHitsProduct, std::move(holder)};
  auto const& PFRecHits = ctx.get(PFRecHitsProduct);
  cudaStream = ctx.stream();
  const Int_t nRH = PFRecHits.size;

  if (initCuda) {
    // Only allocate Cuda memory on first event
    PFClusterCudaHCAL::initializeCudaConstants(cudaConstants, cudaStream);

    inputCPU.allocate(cudaConfig_, cudaStream);
    inputGPU.allocate(cudaConfig_, cudaStream);

    outputCPU.allocate(cudaConfig_, cudaStream);
    outputGPU.allocate(cudaConfig_, cudaStream);
    scratchGPU.allocate(cudaConfig_, cudaStream);

    initCuda = false;
  }

  /* KenH
  _initialClustering->reset();
  if (_pfClusterBuilder)
    _pfClusterBuilder->reset();
  */

  //#ifdef DEBUG_WithLEGACY
  edm::Handle<reco::PFRecHitCollection> rechits;
  if (_produceLegacy)
    event.getByToken(_rechitsLabel, rechits);
  //#endif

  /* KenH
  _initialClustering->updateEvent(event);
  */

  const int numbytes_int = nRH * sizeof(int);
  int totalNeighbours = 0;  // Running count of 8 neighbour edges for edgeId, edgeList

  float kernelTimers[8] = {0.0};

  if (cudaStreamQuery(cudaStream) != cudaSuccess)
    cudaCheck(cudaStreamSynchronize(cudaStream));

  // Calling cuda kernels
  PFClusterCudaHCAL::PFRechitToPFCluster_HCAL_entryPoint(
      cudaStream, (int)totalNeighbours, PFRecHits, inputGPU, outputCPU, outputGPU, scratchGPU, kernelTimers);

  // Data transfer from GPU
  if (cudaStreamQuery(cudaStream) != cudaSuccess)
    cudaCheck(cudaStreamSynchronize(cudaStream));
  cudaCheck(cudaMemcpyAsync(
      outputCPU.topoIter.get(), outputGPU.topoIter.get(), sizeof(int), cudaMemcpyDeviceToHost, cudaStream));
  cudaCheck(cudaMemcpyAsync(
      outputCPU.pcrhFracSize.get(), outputGPU.pcrhFracSize.get(), sizeof(int), cudaMemcpyDeviceToHost, cudaStream));
  cudaCheck(
      cudaMemcpyAsync(outputCPU.nEdges.get(), outputGPU.nEdges.get(), sizeof(int), cudaMemcpyDeviceToHost, cudaStream));
  if (cudaStreamQuery(cudaStream) != cudaSuccess)
    cudaCheck(cudaStreamSynchronize(cudaStream));

  // Total size of allocated rechit fraction arrays (includes some extra padding for rechits that don't end up passing cuts)
  const Int_t nFracs = outputCPU.pcrhFracSize[0];

  cudaCheck(cudaMemcpyAsync(
      outputCPU.pfc_iter.get(), outputGPU.pfc_iter.get(), numbytes_int, cudaMemcpyDeviceToHost, cudaStream));
  cudaCheck(cudaMemcpyAsync(
      outputCPU.topoSeedCount.get(), outputGPU.topoSeedCount.get(), numbytes_int, cudaMemcpyDeviceToHost, cudaStream));
  cudaCheck(cudaMemcpyAsync(
      outputCPU.topoRHCount.get(), outputGPU.topoRHCount.get(), numbytes_int, cudaMemcpyDeviceToHost, cudaStream));
  cudaCheck(cudaMemcpyAsync(outputCPU.seedFracOffsets.get(),
                            outputGPU.seedFracOffsets.get(),
                            numbytes_int,
                            cudaMemcpyDeviceToHost,
                            cudaStream));
  cudaCheck(cudaMemcpyAsync(outputCPU.topoSeedOffsets.get(),
                            outputGPU.topoSeedOffsets.get(),
                            numbytes_int,
                            cudaMemcpyDeviceToHost,
                            cudaStream));
  cudaCheck(cudaMemcpyAsync(
      outputCPU.topoSeedList.get(), outputGPU.topoSeedList.get(), numbytes_int, cudaMemcpyDeviceToHost, cudaStream));

  cudaCheck(cudaMemcpyAsync(outputCPU.pcrh_fracInd.get(),
                            outputGPU.pcrh_fracInd.get(),
                            sizeof(int) * nFracs,
                            cudaMemcpyDeviceToHost,
                            cudaStream));
  cudaCheck(cudaMemcpyAsync(
      outputCPU.pcrh_frac.get(), outputGPU.pcrh_frac.get(), sizeof(int) * nFracs, cudaMemcpyDeviceToHost, cudaStream));
  cudaCheck(cudaMemcpyAsync(
      outputCPU.pfrh_isSeed.get(), outputGPU.pfrh_isSeed.get(), numbytes_int, cudaMemcpyDeviceToHost, cudaStream));
  cudaCheck(cudaMemcpyAsync(
      outputCPU.pfrh_topoId.get(), outputGPU.pfrh_topoId.get(), numbytes_int, cudaMemcpyDeviceToHost, cudaStream));

  cudaCheck(cudaMemcpyAsync(outputCPU.pfrh_passTopoThresh.get(),
                            outputGPU.pfrh_passTopoThresh.get(),
                            sizeof(int) * nRH,
                            cudaMemcpyDeviceToHost,
                            cudaStream));

  if (cudaStreamQuery(cudaStream) != cudaSuccess)
    cudaCheck(cudaStreamSynchronize(cudaStream));

  //
  // Build PFClusters in legacy format

  std::unordered_map<int, std::vector<int>> nTopoRechits;
  std::unordered_map<int, int> nTopoSeeds;

  //auto pfClustersFromCuda = std::make_unique<reco::PFClusterCollection>();
  pfClustersFromCuda = std::make_unique<reco::PFClusterCollection>();

  if (_produceLegacy) {
  for (int rh = 0; rh < nRH; rh++) {
    int topoId = outputCPU.pfrh_topoId[rh];
    if (topoId > -1) {
      // Valid topo id
      nTopoRechits[topoId].push_back(rh);
      if (outputCPU.pfrh_isSeed[rh] > 0) {
        nTopoSeeds[topoId]++;
      }
    }
  }

  // Looping over PFRecHits for creating PFClusters
  for (int n = 0; n < (int)nRH; n++) {
    if (outputCPU.pfrh_isSeed[n] == 1) { // If this PFRecHit is a seed, this should form a PFCluster. Compute necessary information.
      reco::PFCluster temp;
      temp.setSeed((*rechits)[n].detId()); // Pulling the detId of this PFRecHit from the legacy format input
      int offset = outputCPU.seedFracOffsets[n];
      int topoId = outputCPU.pfrh_topoId[n];
      int nSeeds = outputCPU.topoSeedCount[topoId];
      for (int k = offset; k < (offset + outputCPU.topoRHCount[topoId] - nSeeds + 1); k++) { // Looping over PFRecHits in the same topo cluster
        if (outputCPU.pcrh_fracInd[k] > -1 && outputCPU.pcrh_frac[k] > 0.0) {
          const reco::PFRecHitRef& refhit = reco::PFRecHitRef(rechits, outputCPU.pcrh_fracInd[k]);
          temp.addRecHitFraction(reco::PFRecHitFraction(refhit, outputCPU.pcrh_frac[k]));
        }
      }
      // Now PFRedHitFraction of this PFCluster is set. Now compute calculateAndSetPosition (energy, position etc)
      // Check if this topoId has one only one seed
      if (nTopoSeeds.count(outputCPU.pfrh_topoId[n]) && nTopoSeeds[outputCPU.pfrh_topoId[n]] == 1 &&
          _allCellsPositionCalc) {
        _allCellsPositionCalc->calculateAndSetPosition(temp);
      } else {
        _positionCalc->calculateAndSetPosition(temp);
      }
      pfClustersFromCuda->insert(pfClustersFromCuda->end(), std::move(temp));
    }
  }
  } // _produceLegacy

}

void PFClusterProducerCudaHCAL::produce(edm::Event& event, const edm::EventSetup& setup) {
  // cms::cuda::ScopedContextProduce ctx{cudaState_};
  // if (_produceSoA)
  //   ctx.emplace(event, OutputPFRecHitSoA_Token_, std::move(outputGPU.PFClusters)); // SoA "PFClusters" still need to be defined.
  if (_produceLegacy)
    event.put(std::move(pfClustersFromCuda));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFClusterProducerCudaHCAL);
