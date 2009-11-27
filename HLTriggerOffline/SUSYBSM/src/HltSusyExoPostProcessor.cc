#include "HLTriggerOffline/SUSYBSM/interface/HltSusyExoPostProcessor.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include <iostream>
#include <string.h>
#include <iomanip>
#include<fstream>
#include <math.h>


HltSusyExoPostProcessor::HltSusyExoPostProcessor(const edm::ParameterSet& pset)
{
  subDir_ = pset.getUntrackedParameter<std::string>("subDir");
}


void HltSusyExoPostProcessor::endRun(edm::Run const& run, edm::EventSetup const& es)
{

  LogDebug("HltSusyExoPostProcessor") << "Start endRun";
  //////////////////////////////////
  // setup DQM stor               //
  //////////////////////////////////
  
  DQMStore * dqm = 0;
  dqm = edm::Service<DQMStore>().operator->();

  if ( ! dqm ) {
    edm::LogInfo("HltSusyExoPostProcessor") << "Cannot create DQMStore instance\n";
    return;
  }

  LogDebug("HltSusyExoPostProcessor") << "DQMStore opened";

  //go to the directory to be processed
  if(dqm->dirExists(subDir_)) dqm->cd(subDir_);
  else {
    edm::LogWarning("HltSusyExoPostProcessor") << "cannot find directory: " << subDir_ << " , skipping";
    return;
  }

  LogDebug("HltSusyExoPostProcessor") << "DQMStore cd";

  // --- set the names in the dbe folders ---
  std::string triggerBitsDir = "/TriggerBits";
  std::string recoSelBitsDir = "/RecoSelection"; 
  std::string mcSelBitsDir = "/McSelection";      




  //get the number of bins of the MonitorElements (valid for all the MonitorElements, independently of selection on the final state)
  //we take the number of bins from the MonitorElements of the source, and we subtract one because the last bin was reserved for the Total number of events
  int nL1bins  = (dqm->get(dqm->pwd() + triggerBitsDir + "/L1Bits")->getNbinsX()); 
  int nHltbins = (dqm->get(dqm->pwd() + triggerBitsDir + "/HltBits")->getNbinsX()); 


  LogDebug("HltSusyExoPostProcessor") << "number of L1 bins = " << nL1bins << " number of HLT bins = " << nHltbins;
  float value = 0;
  float error = 0;

  //Calculate the efficiencies for histos without any selection
  dqm->setCurrentFolder(subDir_ + triggerBitsDir);
  //book the MonitorElements for the efficiencies 
  MonitorElement* hL1EffBeforeCuts  = dqm->book1D("L1Eff", dqm->get(subDir_ + triggerBitsDir + "/L1Paths")->getTH1F());    
  hL1EffBeforeCuts->setTitle("L1 Efficiencies");
  MonitorElement* hHltEffBeforeCuts = dqm->book1D("HltEff",dqm->get(subDir_ + triggerBitsDir + "/HltPaths")->getTH1F());
  hHltEffBeforeCuts->setTitle("HLT Efficiencies");

  LogDebug("HltSusyExoPostProcessor") << "MonitorElements booked";

  //get the total number of events 
  float nTotalBeforeCuts = dqm->get(subDir_ + triggerBitsDir + "/L1Bits")->getBinContent(nL1bins);
  LogDebug("HltSusyExoPostProcessor") << "Total number of events = " << nTotalBeforeCuts;

  //fill the eff histo
  for(int i=0; i<nL1bins; i++) {
    value = (float)dqm->get(subDir_ + triggerBitsDir + "/L1Paths")->getBinContent(i+1) / nTotalBeforeCuts;
    error = sqrt(value*(1-value)/nTotalBeforeCuts);
    hL1EffBeforeCuts->setBinContent(i+1,value);
    hL1EffBeforeCuts->setBinError(i+1,error);
  }
  for(int i=0; i<nHltbins; i++) {
    value = (float)dqm->get(subDir_ + triggerBitsDir + "/HltPaths")->getBinContent(i+1) / nTotalBeforeCuts;
    error = sqrt(value*(1-value)/nTotalBeforeCuts);
    hHltEffBeforeCuts->setBinContent(i+1,value);
    hHltEffBeforeCuts->setBinError(i+1,error);
  }
  LogDebug("HltSusyExoPostProcessor") << "MonitorElements filled";




  //Calculate the efficiencies for histos after MC selection
  dqm->setCurrentFolder(subDir_ + mcSelBitsDir);
  //book the MonitorElements for the efficiencies 
  MonitorElement* hL1EffAfterMcCuts  = dqm->book1D("L1Eff", dqm->get(subDir_ + mcSelBitsDir + "/L1Paths")->getTH1F());    
  hL1EffAfterMcCuts->setTitle("L1 Efficiencies");
  MonitorElement* hHltEffAfterMcCuts = dqm->book1D("HltEff",dqm->get(subDir_ + mcSelBitsDir + "/HltPaths")->getTH1F());
  hHltEffAfterMcCuts->setTitle("HLT Efficiencies");

  LogDebug("HltSusyExoPostProcessor") << "MonitorElements booked";

  //get the total number of events 
  float nTotalAfterMcCuts = dqm->get(subDir_ + mcSelBitsDir + "/L1Bits")->getBinContent(nL1bins);
  LogDebug("HltSusyExoPostProcessor") << "Total number of events = " << nTotalAfterMcCuts;

  //fill the eff histo
  for(int i=0; i<nL1bins; i++) {
    value = (float)dqm->get(subDir_ + mcSelBitsDir + "/L1Paths")->getBinContent(i+1) / nTotalAfterMcCuts;
    error = sqrt(value*(1-value)/nTotalAfterMcCuts);
    hL1EffAfterMcCuts->setBinContent(i+1,value);
    hL1EffAfterMcCuts->setBinError(i+1,error);
  }
  for(int i=0; i<nHltbins; i++) {
    value = (float)dqm->get(subDir_ + mcSelBitsDir + "/HltPaths")->getBinContent(i+1) / nTotalAfterMcCuts;
    error = sqrt(value*(1-value)/nTotalAfterMcCuts);
    hHltEffAfterMcCuts->setBinContent(i+1,value);
    hHltEffAfterMcCuts->setBinError(i+1,error);
  }
  LogDebug("HltSusyExoPostProcessor") << "MonitorElements filled";






  //Calculate the efficiencies for histos after RECO selection
  dqm->setCurrentFolder(subDir_ + recoSelBitsDir);
  //book the MonitorElements for the efficiencies 
  MonitorElement* hL1EffAfterRecoCuts  = dqm->book1D("L1Eff", dqm->get(subDir_ + recoSelBitsDir + "/L1Paths")->getTH1F());    
  hL1EffAfterRecoCuts->setTitle("L1 Efficiencies");
  MonitorElement* hHltEffAfterRecoCuts = dqm->book1D("HltEff",dqm->get(subDir_ + recoSelBitsDir + "/HltPaths")->getTH1F());
  hHltEffAfterRecoCuts->setTitle("HLT Efficiencies");

  LogDebug("HltSusyExoPostProcessor") << "MonitorElements booked";

  //get the total number of events 
  float nTotalAfterRecoCuts = dqm->get(subDir_ + recoSelBitsDir + "/L1Bits")->getBinContent(nL1bins);
  LogDebug("HltSusyExoPostProcessor") << "Total number of events = " << nTotalAfterRecoCuts;

  //fill the eff histo
  for(int i=0; i<nL1bins; i++) {
    value = (float)dqm->get(subDir_ + recoSelBitsDir + "/L1Paths")->getBinContent(i+1) / nTotalAfterRecoCuts;
    error = sqrt(value*(1-value)/nTotalAfterRecoCuts);
    hL1EffAfterRecoCuts->setBinContent(i+1,value);
    hL1EffAfterRecoCuts->setBinError(i+1,error);
  }
  for(int i=0; i<nHltbins; i++) {
    value = (float)dqm->get(subDir_ + recoSelBitsDir + "/HltPaths")->getBinContent(i+1) / nTotalAfterRecoCuts;
    error = sqrt(value*(1-value)/nTotalAfterRecoCuts);
    hHltEffAfterRecoCuts->setBinContent(i+1,value);
    hHltEffAfterRecoCuts->setBinError(i+1,error);
  }
  LogDebug("HltSusyExoPostProcessor") << "MonitorElements filled";




}

DEFINE_FWK_MODULE(HltSusyExoPostProcessor);

  

