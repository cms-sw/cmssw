#include "L1Trigger/TrackFindingTracklet/interface/Setup.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/L1TrackTrigger/interface/TTBV.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>
#include <set>

namespace trklet {

  Setup::Setup(const Config& config, const trackerDTC::Setup* setup) : dtc_(setup), config_(config) {
    numFrames_ = dtc_->sysNumFrames() + dtc_->sysNumFramesInfra() - 1;
    tbMaxCot_ = std::sinh(dtc_->regMaxEta());
    const Settings settings;
    tbBaseInv2R_ = .5 * settings.krinvpars();
    tbBasePhi0_ = settings.kphi0pars();
    tbBaseCot_ = settings.ktpars();
    tbBaseZ0_ = settings.kz0pars();
    tbBaseR_ = settings.kr();
    tbBasePhi_ = settings.kphi1();
    tbBaseZ_ = settings.kz();
    tbBasePhis_.reserve(N_LAYER + N_DISK);
    for (int id = 0; id < N_LAYER + N_DISK; id++)
      tbBasePhis_.push_back(settings.kphi(id));
    tbBaseZs_.reserve(N_LAYER);
    for (int layer = 0; layer < N_LAYER; layer++)
      tbBaseZs_.push_back(settings.kz(layer));
    tbWidthZ_ = settings.zresidbits();
    tbWidthR_ = settings.rresidbits();
    tbWidthPhi_ = settings.phiresidbits();
    tbWidthSeedType_ = tt::ilog2(config_.tbNumSeedTypes);
    tmNumLayers_ = config_.tbNumLayers;
    int maxProjLayers(0);
    for (const std::vector<int>& projLayers : config_.tbSeedTypesProjectionLayers)
      maxProjLayers = std::max(maxProjLayers, static_cast<int>(projLayers.size()));
    drNumLayers_ = config_.tbNumSeedingLayers + maxProjLayers;
    const int baseShiftInvCot = tt::ilog2(config_.tbMaxR / config_.tbMinZ) - dtc_->widthDSPbu();
    drBaseInvCot_ = std::pow(2, baseShiftInvCot);
    kfNumProj_ = config_.kfNumLayers - config_.tbNumSeedingLayers;
    kfMinProj_ = config_.kfMinLayers - config_.tbNumSeedingLayers;
    tmMuxOrder_.reserve(config_.tmMuxOrder.size());
    for (const std::string& seedType : config_.tmMuxOrder) {
      const auto it = std::find(config_.tbSeedTypes.begin(), config_.tbSeedTypes.end(), seedType);
      tmMuxOrder_.push_back(std::distance(config_.tbSeedTypes.begin(), it));
    }
  }

  // returns radial position and phi, z residuals of a TB stub
  GlobalPoint Setup::stubPosTB(const tt::FrameStub& frame, double cot) const {
    // sennsor module for this seed stub
    const trackerDTC::SensorModule* sm = dtc_->sensorModule(frame.first);
    const trackerDTC::SensorModule::Type type = sm->type();
    const int layerIndex = sm->layerIndex();
    const bool barrel = sm->barrel();
    const int widthR = config_.tbWidthsR[type];
    const int widthRZ = barrel ? tbWidthZ_ : tbWidthR_;
    const double baseR = dtc_->stubBaseR(type);
    const double basePhi = barrel ? tbBasePhi_ : tbBasePhis_[sm->layerIndexCombined()];
    const double baseRZ = barrel ? tbBaseZs_[layerIndex] : tbBaseZ_;
    // parse residuals from tt::Frame and take layerId from tt::TTStubRef
    TTBV bv(frame.second);
    const double z = bv.val(widthRZ, 0, true) * baseRZ * (barrel ? 1. : -cot);
    bv >>= widthRZ;
    const double phi = bv.val(tbWidthPhi_, 0, true) * basePhi;
    bv >>= tbWidthPhi_;
    double r = bv.val(widthR, 0, barrel) * baseR + (barrel ? tt::digiR(dtc_->stubLayerR(layerIndex), tbBaseR_) : 0.0);
    if (type == trackerDTC::SensorModule::Disk2S)
      r = tt::digiR(dtc_->stubDiskR(layerIndex, r), tbBaseR_);
    return GlobalPoint(GlobalPoint::Cylindrical(r, phi, z));
  }

  // returns radial position and phi, z residuals of a TB fake seed stub
  GlobalPoint Setup::stubPosTB(const TTStubRef& ttStubRef, double cot, double z0) const {
    // sennsor module for this seed stub
    const trackerDTC::SensorModule* sm = dtc_->sensorModule(ttStubRef);
    const int layerIndex = sm->layerIndex();
    double r;
    if (sm->barrel()) {
      r = tt::digiR(dtc_->stubLayerR(layerIndex), tbBaseR_);
      const double z = tt::digi(z0 + r * cot, tbBaseZ_);
      if (std::abs(z) > tt::digi(config_.tbMinZ, tbBaseZ_) && layerIndex == 0)
        r = tt::digiR(config_.tbInnerRadius, tbBaseR_);
    } else {
      const double z = tt::digi((sm->side() ? 1. : -1.) * dtc_->stubDiskZ(layerIndex), tbBaseZ0_);
      const double invCot = tt::digi(1. / std::abs(cot), drBaseInvCot_);
      r = tt::digiR((z - z0) * invCot, tbBaseR_);
    }
    return GlobalPoint(GlobalPoint::Cylindrical(r, 0, 0));
  }

}  // namespace trklet
