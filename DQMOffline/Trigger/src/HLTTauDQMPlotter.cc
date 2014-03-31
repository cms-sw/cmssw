#include "DQMOffline/Trigger/interface/HLTTauDQMPlotter.h"

#include "Math/GenVector/VectorUtil.h"

HLTTauDQMPlotter::HLTTauDQMPlotter(const edm::ParameterSet& pset, const std::string& dqmBaseFolder):
  dqmFullFolder_(dqmBaseFolder),
  configValid_(false),
  runValid_(false)
{
  try {
    dqmFolder_ = pset.getUntrackedParameter<std::string>("DQMFolder");
    dqmFullFolder_ += dqmFolder_;
    configValid_  = true;
  } catch ( cms::Exception &e ) {
    edm::LogWarning("HLTTauDQMOfflineSource") << "HLTTauDQMPlotter::HLTTauDQMPlotter(): " << e.what();
    configValid_ = false;
  }
}

HLTTauDQMPlotter::HLTTauDQMPlotter(const std::string& dqmFolder, const std::string& dqmBaseFolder):
  dqmFullFolder_(dqmBaseFolder+dqmFolder),
  dqmFolder_(dqmFolder),
  configValid_(true),
  runValid_(false)
{}

HLTTauDQMPlotter::~HLTTauDQMPlotter() {
}

std::pair<bool,LV> HLTTauDQMPlotter::match( const LV& jet, const LVColl& McInfo, double dr ) {
    bool matched = false;
    LV out;
    for ( std::vector<LV>::const_iterator it = McInfo.begin(); it != McInfo.end(); ++it ) {
        double delta = ROOT::Math::VectorUtil::DeltaR(jet,*it);
        if ( delta < dr ) {
            matched = true;
            out = *it;
            break;
        }
    }
    return std::pair<bool,LV>(matched,out);
}
