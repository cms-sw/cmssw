#include "DQM/EcalCommon/interface/EcalMEFormatter.h"

#include "DQM/EcalCommon/interface/MESetDet2D.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include <limits>

EcalMEFormatter::EcalMEFormatter(edm::ParameterSet const& _ps) :
  edm::EDAnalyzer(),
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
EcalMEFormatter::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
  format_(true);
}

void
EcalMEFormatter::endRun(edm::Run const&, edm::EventSetup const&)
{
  format_(false);
}

void
EcalMEFormatter::format_(bool _checkLumi)
{
  DQMStore& dqmStore(*edm::Service<DQMStore>());
  std::string failedPath;
    
  for(ecaldqm::MESetCollection::iterator mItr(MEs_.begin()); mItr != MEs_.end(); ++mItr){
    if(_checkLumi && !mItr->second->getLumiFlag()) continue;
    mItr->second->clear();
    if(!mItr->second->retrieve(dqmStore, &failedPath)){
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
  if(_meSet.getKind() != MonitorElement::DQM_KIND_TPROFILE2D) return;

  MonitorElement* me(0);
  unsigned iME(0);
  while((me = _meSet.getME(iME++))){
    TProfile2D* prof(me->getTProfile2D());
    for(int iX(1); iX <= prof->GetNbinsX(); ++iX){
      for(int iY(1); iY <= prof->GetNbinsY(); ++iY){
        int bin(prof->GetBin(iX, iY));
        if(prof->GetBinEntries(bin) == 0.){
          if(verbosity_ > 2) edm::LogInfo("EcalDQM") << "Found empty bin " << bin << " in histogram " << prof->GetName();
          // TEMPORARY SETUP UNTIL RENDERPLUGIN IS UPDATED TO DO THIS
          // WHEN IT IS, SWITCH TO ENTRIES -1 CONTENT 0
          prof->SetBinEntries(bin, 1.);
          prof->SetBinContent(bin, -std::numeric_limits<double>::max());
        }
      }
    }
  }
}

DEFINE_FWK_MODULE(EcalMEFormatter);
