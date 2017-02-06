#include "DQM/EcalCommon/interface/EcalMEFormatter.h"

#include "DQM/EcalCommon/interface/MESetDet2D.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include <limits>

EcalMEFormatter::EcalMEFormatter(edm::ParameterSet const& _ps) :
  DQMEDHarvester(),
  ecaldqm::DQWorker()
{
  initialize("EcalMEFormatter", _ps);
  setME(_ps.getUntrackedParameterSet("MEs"));
  verbosity_ = _ps.getUntrackedParameter<int>("verbosity", 0);
}

/*static*/
void
EcalMEFormatter::fillDescriptions(edm::ConfigurationDescriptions& _descs)
{
  edm::ParameterSetDescription desc;
  ecaldqm::DQWorker::fillDescriptions(desc);
  desc.addUntracked<int>("verbosity", 0);

  _descs.addDefault(desc);
}

void
EcalMEFormatter::dqmEndLuminosityBlock(DQMStore::IBooker&, DQMStore::IGetter& _igetter, edm::LuminosityBlock const&, edm::EventSetup const&)
{
  format_(_igetter, true);
}

void
EcalMEFormatter::dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter& _igetter)
{
  format_(_igetter, false);
}

void
EcalMEFormatter::format_(DQMStore::IGetter& _igetter, bool _checkLumi)
{
  std::string failedPath;
    
  for(ecaldqm::MESetCollection::iterator mItr(MEs_.begin()); mItr != MEs_.end(); ++mItr){
    if(_checkLumi && !mItr->second->getLumiFlag()) continue;
    mItr->second->clear();
    if(!mItr->second->retrieve(_igetter, &failedPath)){
      if(verbosity_ > 0) edm::LogWarning("EcalDQM") << "Could not find ME " << mItr->first << "@" << failedPath;
      continue;
    }
    if(verbosity_ > 1) edm::LogInfo("EcalDQM") << "Retrieved " << mItr->first << " from DQMStore";

    if(dynamic_cast<ecaldqm::MESetDet2D*>(mItr->second)) formatDet2D_(*mItr->second);
  }
}

void
EcalMEFormatter::formatDet2D_(ecaldqm::MESet& _meSet)
{
  return;
}

DEFINE_FWK_MODULE(EcalMEFormatter);
