#include <utility>

#include "DQMOffline/Trigger/interface/HLTTauDQMPlotter.h"

#include "Math/GenVector/VectorUtil.h"

HLTTauDQMPlotter::HLTTauDQMPlotter(const edm::ParameterSet& pset, std::string  dqmBaseFolder):
  dqmFullFolder_(std::move(dqmBaseFolder)),
  configValid_(false)
{
  dqmFolder_ = pset.getUntrackedParameter<std::string>("DQMFolder");
  dqmFullFolder_ += "/";
  dqmFullFolder_ += dqmFolder_;
  configValid_  = true;
}

HLTTauDQMPlotter::HLTTauDQMPlotter(const std::string& dqmFolder, const std::string& dqmBaseFolder):
  dqmFullFolder_(dqmBaseFolder+"/"+dqmFolder),
  dqmFolder_(dqmFolder),
  configValid_(true)
{}

HLTTauDQMPlotter::~HLTTauDQMPlotter() = default;

std::pair<bool,LV> HLTTauDQMPlotter::match( const LV& jet, const LVColl& McInfo, double dr ) {
    bool matched = false;
    LV out;
    for (auto const & it : McInfo) {
        double delta = ROOT::Math::VectorUtil::DeltaR(jet,it);
        if ( delta < dr ) {
            matched = true;
            out = it;
            break;
        }
    }
    return std::pair<bool,LV>(matched,out);
}
