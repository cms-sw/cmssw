#include "L1Trigger/TrackerTFP/interface/Setup.h"
#include "DataFormats/L1TrackTrigger/interface/TTBV.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>
#include <set>

namespace trackerTFP {

  Setup::Setup(const Config& config, const trackerDTC::Setup* setup) : dtc_(setup), config_(config) {
    // gp
    maxRphi_ = std::max(std::abs(dtc_->sysOuterRadius() - dtc_->regChosenRofPhi()),
                        std::abs(dtc_->sysInnerRadius() - dtc_->regChosenRofPhi()));
    maxRz_ = std::max(std::abs(dtc_->sysOuterRadius() - dtc_->regChosenRofZ()),
                      std::abs(dtc_->sysInnerRadius() - dtc_->regChosenRofZ()));
    gpNumSector_ = config_.gpNumBinsPhiT * config_.gpNumBinsZT;
    // ctb
    ctbWidthLayerCount_ = std::ceil(std::log2(config_.ctbMaxStubs));
    // kf
  }

}  // namespace trackerTFP
