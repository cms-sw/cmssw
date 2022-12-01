#include <cuda_runtime.h>

#include "CondFormats/SiPixelTransient/interface/SiPixelTemplate.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFast.h"

// Services
// this is needed to get errors from templates

namespace {
  constexpr float micronsToCm = 1.0e-4;
}

//-----------------------------------------------------------------------------
//!  The constructor.
//-----------------------------------------------------------------------------
template <typename TrackerTraits>
PixelCPEFast<TrackerTraits>::PixelCPEFast(edm::ParameterSet const& conf,
                                          const MagneticField* mag,
                                          const TrackerGeometry& geom,
                                          const TrackerTopology& ttopo,
                                          const SiPixelLorentzAngle* lorentzAngle,
                                          const SiPixelGenErrorDBObject* genErrorDBObject,
                                          const SiPixelLorentzAngle* lorentzAngleWidth)
    : PixelCPEGenericBase(conf, mag, geom, ttopo, lorentzAngle, genErrorDBObject, lorentzAngleWidth) {
  // Use errors from templates or from GenError
  if (useErrorsFromTemplates_) {
    if (!SiPixelGenError::pushfile(*genErrorDBObject_, thePixelGenError_))
      throw cms::Exception("InvalidCalibrationLoaded")
          << "ERROR: GenErrors not filled correctly. Check the sqlite file. Using SiPixelTemplateDBObject version "
          << (*genErrorDBObject_).version();
  }

  fillParamsForGpu();

  cpuData_ = {
      &commonParamsGPU_,
      detParamsGPU_.data(),
      &layerGeometry_,
      &averageGeometry_,
  };
}

template <typename TrackerTraits>
const pixelCPEforGPU::ParamsOnGPUT<TrackerTraits>* PixelCPEFast<TrackerTraits>::getGPUProductAsync(
    cudaStream_t cudaStream) const {
  using ParamsOnGPU = pixelCPEforGPU::ParamsOnGPUT<TrackerTraits>;
  using LayerGeometry = pixelCPEforGPU::LayerGeometryT<TrackerTraits>;
  using AverageGeometry = pixelTopology::AverageGeometryT<TrackerTraits>;

  const auto& data = gpuData_.dataForCurrentDeviceAsync(cudaStream, [this](GPUData& data, cudaStream_t stream) {
    // and now copy to device...

    cudaCheck(cudaMalloc((void**)&data.paramsOnGPU_h.m_commonParams, sizeof(pixelCPEforGPU::CommonParams)));
    cudaCheck(cudaMalloc((void**)&data.paramsOnGPU_h.m_detParams,
                         this->detParamsGPU_.size() * sizeof(pixelCPEforGPU::DetParams)));
    cudaCheck(cudaMalloc((void**)&data.paramsOnGPU_h.m_averageGeometry, sizeof(AverageGeometry)));
    cudaCheck(cudaMalloc((void**)&data.paramsOnGPU_h.m_layerGeometry, sizeof(LayerGeometry)));
    cudaCheck(cudaMalloc((void**)&data.paramsOnGPU_d, sizeof(ParamsOnGPU)));
    cudaCheck(cudaMemcpyAsync(data.paramsOnGPU_d, &data.paramsOnGPU_h, sizeof(ParamsOnGPU), cudaMemcpyDefault, stream));
    cudaCheck(cudaMemcpyAsync((void*)data.paramsOnGPU_h.m_commonParams,
                              &this->commonParamsGPU_,
                              sizeof(pixelCPEforGPU::CommonParams),
                              cudaMemcpyDefault,
                              stream));
    cudaCheck(cudaMemcpyAsync((void*)data.paramsOnGPU_h.m_averageGeometry,
                              &this->averageGeometry_,
                              sizeof(AverageGeometry),
                              cudaMemcpyDefault,
                              stream));
    cudaCheck(cudaMemcpyAsync((void*)data.paramsOnGPU_h.m_layerGeometry,
                              &this->layerGeometry_,
                              sizeof(LayerGeometry),
                              cudaMemcpyDefault,
                              stream));
    cudaCheck(cudaMemcpyAsync((void*)data.paramsOnGPU_h.m_detParams,
                              this->detParamsGPU_.data(),
                              this->detParamsGPU_.size() * sizeof(pixelCPEforGPU::DetParams),
                              cudaMemcpyDefault,
                              stream));
  });
  return data.paramsOnGPU_d;
}

template <typename TrackerTraits>
void PixelCPEFast<TrackerTraits>::fillParamsForGpu() {
  //
  // this code executes only once per job, computation inefficiency is not an issue
  // many code blocks are repeated: better keep the computation local and self oconsistent as blocks may in future move around, be deleted ...
  // It is valid only for Phase1 and the version of GenError in DB used in late 2018 and in 2021

  commonParamsGPU_.theThicknessB = m_DetParams.front().theThickness;
  commonParamsGPU_.theThicknessE = m_DetParams.back().theThickness;
  commonParamsGPU_.thePitchX = m_DetParams[0].thePitchX;
  commonParamsGPU_.thePitchY = m_DetParams[0].thePitchY;

  commonParamsGPU_.numberOfLaddersInBarrel = TrackerTraits::numberOfLaddersInBarrel;

  LogDebug("PixelCPEFast") << "pitch & thickness " << commonParamsGPU_.thePitchX << ' ' << commonParamsGPU_.thePitchY
                           << "  " << commonParamsGPU_.theThicknessB << ' ' << commonParamsGPU_.theThicknessE;

  // zero average geometry
  memset(&averageGeometry_, 0, sizeof(pixelTopology::AverageGeometryT<TrackerTraits>));

  uint32_t oldLayer = 0;
  uint32_t oldLadder = 0;
  float rl = 0;
  float zl = 0;
  float miz = 500, mxz = 0;
  float pl = 0;
  int nl = 0;
  detParamsGPU_.resize(m_DetParams.size());

  for (auto i = 0U; i < m_DetParams.size(); ++i) {
    auto& p = m_DetParams[i];
    auto& g = detParamsGPU_[i];

    g.nRowsRoc = p.theDet->specificTopology().rowsperroc();
    g.nColsRoc = p.theDet->specificTopology().colsperroc();
    g.nRows = p.theDet->specificTopology().rocsX() * g.nRowsRoc;
    g.nCols = p.theDet->specificTopology().rocsY() * g.nColsRoc;

    g.numPixsInModule = g.nRows * g.nCols;

    assert(p.theDet->index() == int(i));
    assert(commonParamsGPU_.thePitchY == p.thePitchY);
    assert(commonParamsGPU_.thePitchX == p.thePitchX);

    g.isBarrel = GeomDetEnumerators::isBarrel(p.thePart);
    g.isPosZ = p.theDet->surface().position().z() > 0;
    g.layer = ttopo_.layer(p.theDet->geographicalId());
    g.index = i;  // better be!
    g.rawId = p.theDet->geographicalId();
    auto thickness = g.isBarrel ? commonParamsGPU_.theThicknessB : commonParamsGPU_.theThicknessE;
    assert(thickness == p.theThickness);

    auto ladder = ttopo_.pxbLadder(p.theDet->geographicalId());
    if (oldLayer != g.layer) {
      oldLayer = g.layer;
      LogDebug("PixelCPEFast") << "new layer at " << i << (g.isBarrel ? " B  " : (g.isPosZ ? " E+ " : " E- "))
                               << g.layer << " starting at " << g.rawId << '\n'
                               << "old layer had " << nl << " ladders";
      nl = 0;
    }
    if (oldLadder != ladder) {
      oldLadder = ladder;
      LogDebug("PixelCPEFast") << "new ladder at " << i << (g.isBarrel ? " B  " : (g.isPosZ ? " E+ " : " E- "))
                               << ladder << " starting at " << g.rawId << '\n'
                               << "old ladder ave z,r,p mz " << zl / 8.f << " " << rl / 8.f << " " << pl / 8.f << ' '
                               << miz << ' ' << mxz;
      rl = 0;
      zl = 0;
      pl = 0;
      miz = 500;
      mxz = 0;
      nl++;
    }

    g.shiftX = 0.5f * p.lorentzShiftInCmX;
    g.shiftY = 0.5f * p.lorentzShiftInCmY;
    g.chargeWidthX = p.lorentzShiftInCmX * p.widthLAFractionX;
    g.chargeWidthY = p.lorentzShiftInCmY * p.widthLAFractionY;

    g.x0 = p.theOrigin.x();
    g.y0 = p.theOrigin.y();
    g.z0 = p.theOrigin.z();

    auto vv = p.theDet->surface().position();
    auto rr = pixelCPEforGPU::Rotation(p.theDet->surface().rotation());
    g.frame = pixelCPEforGPU::Frame(vv.x(), vv.y(), vv.z(), rr);

    zl += vv.z();
    miz = std::min(miz, std::abs(vv.z()));
    mxz = std::max(mxz, std::abs(vv.z()));
    rl += vv.perp();
    pl += vv.phi();  // (not obvious)

    // errors .....
    ClusterParamGeneric cp;

    cp.with_track_angle = false;

    auto lape = p.theDet->localAlignmentError();
    if (lape.invalid())
      lape = LocalError();  // zero....

    g.apeXX = lape.xx();
    g.apeYY = lape.yy();

    auto toMicron = [&](float x) { return std::min(511, int(x * 1.e4f + 0.5f)); };

    // average angle
    auto gvx = p.theOrigin.x() + 40.f * commonParamsGPU_.thePitchX;
    auto gvy = p.theOrigin.y();
    auto gvz = 1.f / p.theOrigin.z();
    //--- Note that the normalization is not required as only the ratio used

    {
      // calculate angles (fed into errorFromTemplates)
      cp.cotalpha = gvx * gvz;
      cp.cotbeta = gvy * gvz;

      errorFromTemplates(p, cp, 20000.);
    }

#ifdef EDM_ML_DEBUG
    auto m = 10000.f;
    for (float qclus = 15000; qclus < 35000; qclus += 15000) {
      errorFromTemplates(p, cp, qclus);
      LogDebug("PixelCPEFast") << i << ' ' << qclus << ' ' << cp.pixmx << ' ' << m * cp.sigmax << ' ' << m * cp.sx1
                               << ' ' << m * cp.sx2 << ' ' << m * cp.sigmay << ' ' << m * cp.sy1 << ' ' << m * cp.sy2;
    }
    LogDebug("PixelCPEFast") << i << ' ' << m * std::sqrt(lape.xx()) << ' ' << m * std::sqrt(lape.yy());
#endif  // EDM_ML_DEBUG

    g.pixmx = std::max(0, cp.pixmx);
    g.sx2 = toMicron(cp.sx2);
    g.sy1 = std::max(21, toMicron(cp.sy1));  // for some angles sy1 is very small
    g.sy2 = std::max(55, toMicron(cp.sy2));  // sometimes sy2 is smaller than others (due to angle?)

    //sample xerr as function of position
    // moduleOffsetX is the definition of TrackerTraits::xOffset,
    // needs to be calculated because for Phase2 the modules are not uniform
    float moduleOffsetX = -(0.5f * float(g.nRows) + TrackerTraits::bigPixXCorrection);
    auto const xoff = moduleOffsetX * commonParamsGPU_.thePitchX;

    for (int ix = 0; ix < CPEFastParametrisation::kNumErrorBins; ++ix) {
      auto x = xoff * (1.f - (0.5f + float(ix)) / 8.f);
      auto gvx = p.theOrigin.x() - x;
      auto gvy = p.theOrigin.y();
      auto gvz = 1.f / p.theOrigin.z();
      cp.cotbeta = gvy * gvz;
      cp.cotalpha = gvx * gvz;
      errorFromTemplates(p, cp, 20000.f);
      g.sigmax[ix] = toMicron(cp.sigmax);
      g.sigmax1[ix] = toMicron(cp.sx1);
      LogDebug("PixelCPEFast") << "sigmax vs x " << i << ' ' << x << ' ' << cp.cotalpha << ' ' << int(g.sigmax[ix])
                               << ' ' << int(g.sigmax1[ix]) << ' ' << 10000.f * cp.sigmay << std::endl;
    }
#ifdef EDM_ML_DEBUG
    // sample yerr as function of position
    // moduleOffsetY is the definition of TrackerTraits::yOffset (removed)
    float moduleOffsetY = 0.5f * float(g.nCols) + TrackerTraits::bigPixYCorrection;
    auto const yoff = -moduleOffsetY * commonParamsGPU_.thePitchY;

    for (int ix = 0; ix < CPEFastParametrisation::kNumErrorBins; ++ix) {
      auto y = yoff * (1.f - (0.5f + float(ix)) / 8.f);
      auto gvx = p.theOrigin.x() + 40.f * commonParamsGPU_.thePitchY;
      auto gvy = p.theOrigin.y() - y;
      auto gvz = 1.f / p.theOrigin.z();
      cp.cotbeta = gvy * gvz;
      cp.cotalpha = gvx * gvz;
      errorFromTemplates(p, cp, 20000.f);
      LogDebug("PixelCPEFast") << "sigmay vs y " << i << ' ' << y << ' ' << cp.cotbeta << ' ' << 10000.f * cp.sigmay
                               << std::endl;
    }
#endif  // EDM_ML_DEBUG

    // calculate angles (repeated)
    cp.cotalpha = gvx * gvz;
    cp.cotbeta = gvy * gvz;
    auto aveCB = cp.cotbeta;

    // sample x by charge
    int qbin = CPEFastParametrisation::kGenErrorQBins;  // low charge
    int k = 0;
    for (int qclus = 1000; qclus < 200000; qclus += 1000) {
      errorFromTemplates(p, cp, qclus);
      if (cp.qBin_ == qbin)
        continue;
      qbin = cp.qBin_;
      g.xfact[k] = cp.sigmax;
      g.yfact[k] = cp.sigmay;
      g.minCh[k++] = qclus;
#ifdef EDM_ML_DEBUG
      LogDebug("PixelCPEFast") << i << ' ' << g.rawId << ' ' << cp.cotalpha << ' ' << qclus << ' ' << cp.qBin_ << ' '
                               << cp.pixmx << ' ' << m * cp.sigmax << ' ' << m * cp.sx1 << ' ' << m * cp.sx2 << ' '
                               << m * cp.sigmay << ' ' << m * cp.sy1 << ' ' << m * cp.sy2 << std::endl;
#endif  // EDM_ML_DEBUG
    }

    assert(k <= CPEFastParametrisation::kGenErrorQBins);

    // fill the rest  (sometimes bin 4 is missing)
    for (int kk = k; kk < CPEFastParametrisation::kGenErrorQBins; ++kk) {
      g.xfact[kk] = g.xfact[k - 1];
      g.yfact[kk] = g.yfact[k - 1];
      g.minCh[kk] = g.minCh[k - 1];
    }
    auto detx = 1.f / g.xfact[0];
    auto dety = 1.f / g.yfact[0];
    for (int kk = 0; kk < CPEFastParametrisation::kGenErrorQBins; ++kk) {
      g.xfact[kk] *= detx;
      g.yfact[kk] *= dety;
    }
    // sample y in "angle"  (estimated from cluster size)
    float ys = 8.f - 4.f;  // apperent bias of half pixel (see plot)
    // plot: https://indico.cern.ch/event/934821/contributions/3974619/attachments/2091853/3515041/DigilessReco.pdf page 25
    // sample yerr as function of "size"
    for (int iy = 0; iy < CPEFastParametrisation::kNumErrorBins; ++iy) {
      ys += 1.f;  // first bin 0 is for size 9  (and size is in fixed point 2^3)
      if (CPEFastParametrisation::kNumErrorBins - 1 == iy)
        ys += 8.f;  // last bin for "overflow"
      // cp.cotalpha = ys*(commonParamsGPU_.thePitchX/(8.f*thickness));  //  use this to print sampling in "x"  (and comment the line below)
      cp.cotbeta = std::copysign(ys * (commonParamsGPU_.thePitchY / (8.f * thickness)), aveCB);
      errorFromTemplates(p, cp, 20000.f);
      g.sigmay[iy] = toMicron(cp.sigmay);
      LogDebug("PixelCPEFast") << "sigmax/sigmay " << i << ' ' << (ys + 4.f) / 8.f << ' ' << cp.cotalpha << '/'
                               << cp.cotbeta << ' ' << 10000.f * cp.sigmax << '/' << int(g.sigmay[iy]) << std::endl;
    }
  }  // loop over det

  constexpr int numberOfModulesInLadder = TrackerTraits::numberOfModulesInLadder;
  constexpr int numberOfLaddersInBarrel = TrackerTraits::numberOfLaddersInBarrel;
  constexpr int numberOfModulesInBarrel = TrackerTraits::numberOfModulesInBarrel;

  constexpr float ladderFactor = 1.f / float(numberOfModulesInLadder);

  constexpr int firstEndcapPos = TrackerTraits::firstEndcapPos;
  constexpr int firstEndcapNeg = TrackerTraits::firstEndcapNeg;

  // compute ladder baricenter (only in global z) for the barrel
  //
  auto& aveGeom = averageGeometry_;
  int il = 0;
  for (int im = 0, nm = numberOfModulesInBarrel; im < nm; ++im) {
    auto const& g = detParamsGPU_[im];
    il = im / numberOfModulesInLadder;
    assert(il < int(numberOfLaddersInBarrel));
    auto z = g.frame.z();
    aveGeom.ladderZ[il] += ladderFactor * z;
    aveGeom.ladderMinZ[il] = std::min(aveGeom.ladderMinZ[il], z);
    aveGeom.ladderMaxZ[il] = std::max(aveGeom.ladderMaxZ[il], z);
    aveGeom.ladderX[il] += ladderFactor * g.frame.x();
    aveGeom.ladderY[il] += ladderFactor * g.frame.y();
    aveGeom.ladderR[il] += ladderFactor * sqrt(g.frame.x() * g.frame.x() + g.frame.y() * g.frame.y());
  }
  assert(il + 1 == int(numberOfLaddersInBarrel));
  // add half_module and tollerance
  constexpr float moduleLength = TrackerTraits::moduleLength;
  constexpr float module_tolerance = 0.2f;
  for (int il = 0, nl = numberOfLaddersInBarrel; il < nl; ++il) {
    aveGeom.ladderMinZ[il] -= (0.5f * moduleLength - module_tolerance);
    aveGeom.ladderMaxZ[il] += (0.5f * moduleLength - module_tolerance);
  }

  // compute "max z" for first layer in endcap (should we restrict to the outermost ring?)
  for (auto im = TrackerTraits::layerStart[firstEndcapPos]; im < TrackerTraits::layerStart[firstEndcapPos + 1]; ++im) {
    auto const& g = detParamsGPU_[im];
    aveGeom.endCapZ[0] = std::max(aveGeom.endCapZ[0], g.frame.z());
  }
  for (auto im = TrackerTraits::layerStart[firstEndcapNeg]; im < TrackerTraits::layerStart[firstEndcapNeg + 1]; ++im) {
    auto const& g = detParamsGPU_[im];
    aveGeom.endCapZ[1] = std::min(aveGeom.endCapZ[1], g.frame.z());
  }
  // correct for outer ring being closer
  aveGeom.endCapZ[0] -= TrackerTraits::endcapCorrection;
  aveGeom.endCapZ[1] += TrackerTraits::endcapCorrection;
#ifdef EDM_ML_DEBUG
  for (int jl = 0, nl = numberOfLaddersInBarrel; jl < nl; ++jl) {
    LogDebug("PixelCPEFast") << jl << ':' << aveGeom.ladderR[jl] << '/'
                             << std::sqrt(aveGeom.ladderX[jl] * aveGeom.ladderX[jl] +
                                          aveGeom.ladderY[jl] * aveGeom.ladderY[jl])
                             << ',' << aveGeom.ladderZ[jl] << ',' << aveGeom.ladderMinZ[jl] << ','
                             << aveGeom.ladderMaxZ[jl] << '\n';
  }
  LogDebug("PixelCPEFast") << aveGeom.endCapZ[0] << ' ' << aveGeom.endCapZ[1];
#endif  // EDM_ML_DEBUG

  // fill Layer and ladders geometry
  memset(&layerGeometry_, 0, sizeof(pixelCPEforGPU::LayerGeometryT<TrackerTraits>));
  memcpy(layerGeometry_.layerStart,
         TrackerTraits::layerStart,
         sizeof(pixelCPEforGPU::LayerGeometryT<TrackerTraits>::layerStart));
  memcpy(layerGeometry_.layer, pixelTopology::layer<TrackerTraits>.data(), pixelTopology::layer<TrackerTraits>.size());
  layerGeometry_.maxModuleStride = pixelTopology::maxModuleStride<TrackerTraits>;
}

template <typename TrackerTraits>
PixelCPEFast<TrackerTraits>::GPUData::~GPUData() {
  if (paramsOnGPU_d != nullptr) {
    cudaFree((void*)paramsOnGPU_h.m_commonParams);
    cudaFree((void*)paramsOnGPU_h.m_detParams);
    cudaFree((void*)paramsOnGPU_h.m_averageGeometry);
    cudaFree((void*)paramsOnGPU_h.m_layerGeometry);
    cudaFree(paramsOnGPU_d);
  }
}

template <typename TrackerTraits>
void PixelCPEFast<TrackerTraits>::errorFromTemplates(DetParam const& theDetParam,
                                                     ClusterParamGeneric& theClusterParam,
                                                     float qclus) const {
  float locBz = theDetParam.bz;
  float locBx = theDetParam.bx;
  LogDebug("PixelCPEFast") << "PixelCPEFast::localPosition(...) : locBz = " << locBz;

  theClusterParam.pixmx = std::numeric_limits<int>::max();  // max pixel charge for truncation of 2-D cluster

  theClusterParam.sigmay = -999.9;  // CPE Generic y-error for multi-pixel cluster
  theClusterParam.sigmax = -999.9;  // CPE Generic x-error for multi-pixel cluster
  theClusterParam.sy1 = -999.9;     // CPE Generic y-error for single single-pixel
  theClusterParam.sy2 = -999.9;     // CPE Generic y-error for single double-pixel cluster
  theClusterParam.sx1 = -999.9;     // CPE Generic x-error for single single-pixel cluster
  theClusterParam.sx2 = -999.9;     // CPE Generic x-error for single double-pixel cluster

  float dummy;

  SiPixelGenError gtempl(thePixelGenError_);
  int gtemplID = theDetParam.detTemplateId;

  theClusterParam.qBin_ = gtempl.qbin(gtemplID,
                                      theClusterParam.cotalpha,
                                      theClusterParam.cotbeta,
                                      locBz,
                                      locBx,
                                      qclus,
                                      false,
                                      theClusterParam.pixmx,
                                      theClusterParam.sigmay,
                                      dummy,
                                      theClusterParam.sigmax,
                                      dummy,
                                      theClusterParam.sy1,
                                      dummy,
                                      theClusterParam.sy2,
                                      dummy,
                                      theClusterParam.sx1,
                                      dummy,
                                      theClusterParam.sx2,
                                      dummy);

  theClusterParam.sigmax = theClusterParam.sigmax * micronsToCm;
  theClusterParam.sx1 = theClusterParam.sx1 * micronsToCm;
  theClusterParam.sx2 = theClusterParam.sx2 * micronsToCm;

  theClusterParam.sigmay = theClusterParam.sigmay * micronsToCm;
  theClusterParam.sy1 = theClusterParam.sy1 * micronsToCm;
  theClusterParam.sy2 = theClusterParam.sy2 * micronsToCm;
}

template <>
void PixelCPEFast<pixelTopology::Phase2>::errorFromTemplates(DetParam const& theDetParam,
                                                             ClusterParamGeneric& theClusterParam,
                                                             float qclus) const {
  theClusterParam.qBin_ = 0.0f;
}

//-----------------------------------------------------------------------------
//! Hit position in the local frame (in cm).  Unlike other CPE's, this
//! one converts everything from the measurement frame (in channel numbers)
//! into the local frame (in centimeters).
//-----------------------------------------------------------------------------
template <typename TrackerTraits>
LocalPoint PixelCPEFast<TrackerTraits>::localPosition(DetParam const& theDetParam,
                                                      ClusterParam& theClusterParamBase) const {
  ClusterParamGeneric& theClusterParam = static_cast<ClusterParamGeneric&>(theClusterParamBase);

  if (useErrorsFromTemplates_) {
    errorFromTemplates(theDetParam, theClusterParam, theClusterParam.theCluster->charge());
  } else {
    theClusterParam.qBin_ = 0;
  }

  int q_f_X;  //!< Q of the first  pixel  in X
  int q_l_X;  //!< Q of the last   pixel  in X
  int q_f_Y;  //!< Q of the first  pixel  in Y
  int q_l_Y;  //!< Q of the last   pixel  in Y
  collect_edge_charges(theClusterParam, q_f_X, q_l_X, q_f_Y, q_l_Y, useErrorsFromTemplates_ && truncatePixelCharge_);

  // do GPU like ...
  pixelCPEforGPU::ClusParams cp;

  cp.minRow[0] = theClusterParam.theCluster->minPixelRow();
  cp.maxRow[0] = theClusterParam.theCluster->maxPixelRow();
  cp.minCol[0] = theClusterParam.theCluster->minPixelCol();
  cp.maxCol[0] = theClusterParam.theCluster->maxPixelCol();

  cp.q_f_X[0] = q_f_X;
  cp.q_l_X[0] = q_l_X;
  cp.q_f_Y[0] = q_f_Y;
  cp.q_l_Y[0] = q_l_Y;

  cp.charge[0] = theClusterParam.theCluster->charge();

  auto ind = theDetParam.theDet->index();
  pixelCPEforGPU::position<TrackerTraits>(commonParamsGPU_, detParamsGPU_[ind], cp, 0);
  auto xPos = cp.xpos[0];
  auto yPos = cp.ypos[0];

  // set the error  (mind ape....)
  pixelCPEforGPU::errorFromDB<TrackerTraits>(commonParamsGPU_, detParamsGPU_[ind], cp, 0);
  theClusterParam.sigmax = cp.xerr[0];
  theClusterParam.sigmay = cp.yerr[0];

  LogDebug("PixelCPEFast") << " in PixelCPEFast:localPosition - pos = " << xPos << " " << yPos << " size "
                           << cp.maxRow[0] - cp.minRow[0] << ' ' << cp.maxCol[0] - cp.minCol[0];

  //--- Now put the two together
  LocalPoint pos_in_local(xPos, yPos);
  return pos_in_local;
}

//==============  INFLATED ERROR AND ERRORS FROM DB BELOW  ================

//-------------------------------------------------------------------------
//  Hit error in the local frame
//-------------------------------------------------------------------------
template <typename TrackerTraits>
LocalError PixelCPEFast<TrackerTraits>::localError(DetParam const& theDetParam,
                                                   ClusterParam& theClusterParamBase) const {
  ClusterParamGeneric& theClusterParam = static_cast<ClusterParamGeneric&>(theClusterParamBase);

  auto xerr = theClusterParam.sigmax;
  auto yerr = theClusterParam.sigmay;

  LogDebug("PixelCPEFast") << " errors  " << xerr << " " << yerr;

  auto xerr_sq = xerr * xerr;
  auto yerr_sq = yerr * yerr;

  return LocalError(xerr_sq, 0, yerr_sq);
}

template <typename TrackerTraits>
void PixelCPEFast<TrackerTraits>::fillPSetDescription(edm::ParameterSetDescription& desc) {
  // call PixelCPEGenericBase fillPSetDescription to add common rechit errors
  PixelCPEGenericBase::fillPSetDescription(desc);
}

template class PixelCPEFast<pixelTopology::Phase1>;
template class PixelCPEFast<pixelTopology::Phase2>;
