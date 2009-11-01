#include "HLTriggerOffline/JetMET/interface/JetMETDQMPostProcessor.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include <iostream>
#include <string.h>
#include <iomanip>
#include<fstream>
#include <math.h>


JetMETDQMPostProcessor::JetMETDQMPostProcessor(const edm::ParameterSet& pset)
{
  subDir_ = pset.getUntrackedParameter<std::string>("subDir");
}

void JetMETDQMPostProcessor::endRun(edm::Run const& run, edm::EventSetup const& es)
{
  /*
  //////////////////////////////////
  // setup DQM stor               //
  //////////////////////////////////

  DQMStore * dqm = 0;
  dqm = edm::Service<DQMStore>().operator->();

  if ( ! dqm ) {
    edm::LogInfo("JetMETDQMPostProcessor") << "Cannot create DQMStore instance\n";
    return;
  }


  //go to the directory to be processed
  if(dqm->dirExists(subDir_)) dqm->cd(subDir_);
  else {
   edm::LogWarning("JetMETDQMPostProcessor") << "cannot find directory: " << subDir_ << " , skipping";
    return;
  }

  std::vector<std::string> subdirectories = dqm->getSubdirs();
  for(std::vector<std::string>::iterator dir = subdirectories.begin() ;dir!= subdirectories.end(); dir++ ){
    dqm->cd(*dir);
    
    MonitorElement*_meTurnOnMET = dqm->book1D("_meTurnOnMET","Missing ET Turn-On",100,0,500);
    MonitorElement*_meTurnOnJetPt = dqm->book1D("_meTurnOnJetPt","Jet Pt Turn-On",100,0,500);
    
    //_meTurnOnMET->getTH1F()->Add(dqm->get(dqm->pwd() + "_meGenMETTrg")->getTH1F(),1);
    //_meTurnOnMET->getTH1F()->Sumw2();
    // dqm->get(dqm->pwd() + "_meGenMET")->getTH1F()->Sumw2();
//     _meTurnOnMET->getTH1F()->Divide(_meTurnOnMET->getTH1F(),dqm->get(dqm->pwd() + "_meGenMET")->getTH1F(),1,1,"B");
    
//     _meTurnOnJetPt->getTH1F()->Add(_meGenJetPtTrg->getTH1F(),1);
//     _meTurnOnJetPt->getTH1F()->Sumw2();
//     _meGenJetPt->getTH1F()->Sumw2();
//     _meTurnOnJetPt->getTH1F()->Divide(_meTurnOnJetPt->getTH1F(),_meGenJetPt->getTH1F(),1,1,"B");
    
    dqm->goUp();
  }
  */
}
void JetMETDQMPostProcessor::endJob()
{
  //////////////////////////////////
  // setup DQM stor               //
  //////////////////////////////////

  DQMStore * dqm = 0;
  dqm = edm::Service<DQMStore>().operator->();

  if ( ! dqm ) {
    edm::LogInfo("JetMETDQMPostProcessor") << "Cannot create DQMStore instance\n";
    return;
  }


  //go to the directory to be processed
  if(dqm->dirExists(subDir_)) dqm->cd(subDir_);
  else {
   edm::LogWarning("JetMETDQMPostProcessor") << "cannot find directory: " << subDir_ << " , skipping";
    return;
  }

  std::vector<std::string> subdirectories = dqm->getSubdirs();
  for(std::vector<std::string>::iterator dir = subdirectories.begin() ;dir!= subdirectories.end(); dir++ ){
    dqm->cd(*dir);
    
    MonitorElement*_meTurnOnMET = dqm->book1D("_meTurnOnMET","Missing ET Turn-On",100,0,500);
    MonitorElement*_meTurnOnJetPt = dqm->book1D("_meTurnOnJetPt","Jet Pt Turn-On",100,0,500);
    
    //std::vector<std::string> mes = dqm->getMEs();
    //for(std::vector<std::string>::iterator me = mes.begin() ;me!= mes.end(); me++ )
    //  std::cout <<*me <<std::endl;
    //std::cout <<std::endl;
    
    _meTurnOnMET->getTH1F()->Add(dqm->get(dqm->pwd() + "/_meGenMETTrg")->getTH1F(),1);
    _meTurnOnMET->getTH1F()->Sumw2();
    dqm->get(dqm->pwd() + "/_meGenMET")->getTH1F()->Sumw2();
    _meTurnOnMET->getTH1F()->Divide(_meTurnOnMET->getTH1F(),dqm->get(dqm->pwd() + "/_meGenMET")->getTH1F(),1,1,"B");
    
    _meTurnOnJetPt->getTH1F()->Add(dqm->get(dqm->pwd() + "/_meGenJetPtTrg")->getTH1F(),1);
    _meTurnOnJetPt->getTH1F()->Sumw2();
    dqm->get(dqm->pwd() + "/_meGenJetPt")->getTH1F()->Sumw2();
    _meTurnOnJetPt->getTH1F()->Divide(_meTurnOnJetPt->getTH1F(),dqm->get(dqm->pwd() + "/_meGenJetPt")->getTH1F(),1,1,"B");
    
    dqm->goUp();
  }
}
DEFINE_FWK_MODULE(JetMETDQMPostProcessor);
