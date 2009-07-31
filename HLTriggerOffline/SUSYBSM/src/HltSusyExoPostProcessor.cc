#include "HLTriggerOffline/SUSYBSM/interface/HltSusyExoPostProcessor.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include <iostream>
#include <string.h>
#include <iomanip>
#include <fstream>
#include <math.h>
#include "TH1.h"

using namespace std;
using namespace edm;

HltSusyExoPostProcessor::HltSusyExoPostProcessor(const edm::ParameterSet& pset)
{
  subDir_ = pset.getUntrackedParameter<std::string>("subDir");
  mcFlag = pset.getUntrackedParameter<bool>("mc_flag",true);
}


void HltSusyExoPostProcessor::endRun(edm::Run const& run, edm::EventSetup const& es)
{

  LogDebug("HltSusyExoPostProcessor") << "Start endRun";
  //////////////////////////////////
  // setup DQM store              //
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

  std::vector<int> L1placement; for(int i=0; i<nL1bins-1*mcFlag; ++i) L1placement.push_back(6);
  std::vector<int> Hltplacement; for(int i=0; i<nHltbins-1*mcFlag; ++i) Hltplacement.push_back(7);
  int L1bins[7]; for(unsigned int i=0; i<sizeof(L1bins)/sizeof(L1bins[0]); ++i) L1bins[i]=0;
  int Hltbins[8]; for(unsigned int i=0; i<sizeof(Hltbins)/sizeof(Hltbins[0]); ++i) Hltbins[i]=0;
  string L1search[8] = {"Mu","EG","Jet","ET","TauJet","X",""};
  string L1search3 = "HTT", L1search6 = "Bias";
  string Hltsearch[8] = {"Mu","Ele","Jet","Photon","MET","Tau","X",""};
  string Hltsearch4 = "HT", Hltsearch5 = "BTag", Hltsearch6_1 = "EG", Hltsearch6_2 = "ET", Hltsearch7_1 = "Bias", Hltsearch7_2 = "AlCa";


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
  float nTotalBeforeCuts = -1;
  if(mcFlag)
    {
      nTotalBeforeCuts = dqm->get(subDir_ + triggerBitsDir + "/L1Bits")->getBinContent(nL1bins);
    }  
  else
    {
      for(int i=0; i<nHltbins; ++i)
	{
	  string triggername = ((TH1F*)hHltEffBeforeCuts->getTH1F())->GetXaxis()->GetBinLabel(i+1);
	  string eff_search = "HLT_ZeroBias";
	  if(triggername.find(eff_search)!=string::npos && triggername.size()==eff_search.size())
	    {
	      nTotalBeforeCuts = ((TH1F*)hHltEffBeforeCuts->getTH1F())->GetBinContent(i+1);
	      break;
	    }
	}
      if(nTotalBeforeCuts == -1)
	nTotalBeforeCuts = dqm->get(subDir_ + triggerBitsDir + "/L1Bits")->getBinContent(nL1bins);
    }
  LogDebug("HltSusyExoPostProcessor") << "Total number of events = " << nTotalBeforeCuts;

  //fill the eff histo
  for(int i=0; i<nL1bins; i++) {
    value = (float)dqm->get(subDir_ + triggerBitsDir + "/L1Paths")->getBinContent(i+1) / nTotalBeforeCuts;
    error = sqrt(value*(1-value)/nTotalBeforeCuts);
    hL1EffBeforeCuts->setBinContent(i+1,value);
    hL1EffBeforeCuts->setBinError(i+1,error);
    if(i!=nL1bins)
      {
	string triggername = ((TH1F*)hL1EffBeforeCuts->getTH1F())->GetXaxis()->GetBinLabel(i+1);
	if( triggername.find(L1search6)==string::npos )
	  {
	    for(unsigned int j=0; j<sizeof(L1search)/sizeof(L1search[0])-2; ++j)
	      {
		if( triggername.find(L1search[j])!=string::npos || (j==3 && triggername.find(L1search3)!=string::npos) )
		  {
		    if(L1placement[i]==6)
		      L1placement[i]=j;
		    else if(L1placement[i]==2 && j==4)
		      L1placement[i]=4;
		    else
		      L1placement[i]=5;
		  }
		else if(triggername.size()==0 || triggername.find("Total")!=string::npos)
		  L1placement[i]=-1;
	      }
	  }
      }
  }
  for(unsigned int i=0; i<L1placement.size(); ++i) 
    if(L1placement[i]!=-1)
      ++L1bins[L1placement[i]];
  for(int i=0; i<nHltbins; i++) {
    value = (float)dqm->get(subDir_ + triggerBitsDir + "/HltPaths")->getBinContent(i+1) / nTotalBeforeCuts;
    error = sqrt(value*(1-value)/nTotalBeforeCuts);
    hHltEffBeforeCuts->setBinContent(i+1,value);
    hHltEffBeforeCuts->setBinError(i+1,error);
    if(i!=nHltbins)
      {
	string triggername = ((TH1F*)hHltEffBeforeCuts->getTH1F())->GetXaxis()->GetBinLabel(i+1);
	if( triggername.find(Hltsearch7_1)==string::npos && triggername.find(Hltsearch7_2)==string::npos )
	  {
	    for(unsigned int j=0; j<sizeof(Hltsearch)/sizeof(Hltsearch[0])-2; ++j)
	      {
		if( triggername.find(Hltsearch[j])!=string::npos || (j==4 && triggername.find(Hltsearch4)!=string::npos) || (j==5 && triggername.find(Hltsearch5)!=string::npos) )
		  {
		    if(Hltplacement[i]==7)
		      Hltplacement[i]=j;
		    else if( triggername.find(Hltsearch5)!=string::npos )
		      Hltplacement[i]=5;
		    else
		      Hltplacement[i]=6;
		  }
		else if(triggername.size()==0 || triggername.find("Total")!=string::npos)
		  Hltplacement[i]=-1;
	      }
	    if(Hltplacement[i]>=0 && Hltplacement[i]<=5 && (triggername.find(Hltsearch6_1)!=string::npos || (triggername.find(Hltsearch6_2)!=string::npos && Hltplacement[i]!=4) ))
	      Hltplacement[i]=6;
	  }
      }
  }
  for(unsigned int i=0; i<Hltplacement.size(); ++i) 
    if(Hltplacement[i]!=-1)
      ++Hltbins[Hltplacement[i]];

  LogDebug("HltSusyExoPostProcessor") << "MonitorElements filled";

  //Create the sorted histograms
  dqm->setCurrentFolder(subDir_ + triggerBitsDir);
  MonitorElement* hL1EffSorted[7] = {dqm->book1D("L1_Mu", new TH1F("L1_Mu","Efficiencies of L1 Muon Triggers",L1bins[0],0,L1bins[0])),
				     dqm->book1D("L1_EG", new TH1F("L1_EG","Efficiencies of L1 EG Triggers",L1bins[1],0,L1bins[1])),
				     dqm->book1D("L1_Jet", new TH1F("L1_Jet","Efficiencies of L1 Jet Triggers",L1bins[2],0,L1bins[2])),
				     dqm->book1D("L1_ETM_ETT_HTT", new TH1F("L1_ETM_ETT_HTT","Efficiencies of L1 ETM, ETT, and HTT Triggers",L1bins[3],0,L1bins[3])),
				     dqm->book1D("L1_TauJet", new TH1F("L1_TauJet","Efficiencies of L1 TauJet Triggers",L1bins[4],0,L1bins[4])),
				     dqm->book1D("L1_XTrigger", new TH1F("L1_XTrigger","Efficiencies of L1 Cross Triggers",L1bins[5],0,L1bins[5])),
				     dqm->book1D("L1_Overflow", new TH1F("L1_Overflow","Efficiencies of L1 Unsorted Triggers",L1bins[6],0,L1bins[6])) };
  MonitorElement* hHltEffSorted[8] = {dqm->book1D("Hlt_Mu", new TH1F("Hlt_Mu","Efficiencies of HL Muon Triggers",Hltbins[0],0,Hltbins[0])),
				     dqm->book1D("Hlt_Ele", new TH1F("Hlt_Ele","Efficiencies of HL Electron Triggers",Hltbins[1],0,Hltbins[1])),
				     dqm->book1D("Hlt_Jet", new TH1F("Hlt_Jet","Efficiencies of HL Jet Triggers",Hltbins[2],0,Hltbins[2])),
				     dqm->book1D("Hlt_Photon", new TH1F("Hlt_Photon","Efficiencies of HL Photon Triggers",Hltbins[3],0,Hltbins[3])),
				     dqm->book1D("Hlt_MET_HT", new TH1F("Hlt_MET_HT","Efficiencies of HL MET and HT Triggers",Hltbins[4],0,Hltbins[4])),
				     dqm->book1D("Hlt_Tau_BTag", new TH1F("Hlt_Tau_Btag","Efficiencies of HL Tau and BTag Triggers",Hltbins[5],0,Hltbins[5])),
				     dqm->book1D("Hlt_XTrigger", new TH1F("Hlt_XTrigger","Efficiencies of HL Cross Triggers",Hltbins[6],0,Hltbins[6])),
				     dqm->book1D("Hlt_Overflow", new TH1F("Hlt_Overflow","Efficiencies of HL Unsorted Triggers",Hltbins[7],0,Hltbins[7])) };

  int L1bincounter[8]; for(unsigned int i=0; i<sizeof(L1bincounter)/sizeof(L1bincounter[0]); ++i) L1bincounter[i]=0;
  int Hltbincounter[8]; for(unsigned int i=0; i<sizeof(Hltbincounter)/sizeof(Hltbincounter[0]); ++i) Hltbincounter[i]=0;
  TH1F* hL1_ = (TH1F*)hL1EffBeforeCuts->getTH1F();
  TH1F* hHlt_ = (TH1F*)hHltEffBeforeCuts->getTH1F();
  for(unsigned int i=0; i<L1placement.size(); ++i)
    {
      if(L1placement[i]!=-1)
	{
	  hL1EffSorted[L1placement[i]]->setBinLabel(L1bincounter[L1placement[i]]+1, hL1_->GetXaxis()->GetBinLabel(i+1));
	  hL1EffSorted[L1placement[i]]->setBinContent(L1bincounter[L1placement[i]]+1, hL1_->GetBinContent(i+1));
	  hL1EffSorted[L1placement[i]]->setBinError(L1bincounter[L1placement[i]]+1, hL1_->GetBinError(i+1));
	  ++L1bincounter[L1placement[i]];
	}
    }
  for(unsigned int i=0; i<Hltplacement.size(); ++i)
    {
      if(Hltplacement[i]!=-1)
	{
	  hHltEffSorted[Hltplacement[i]]->setBinLabel(Hltbincounter[Hltplacement[i]]+1, hHlt_->GetXaxis()->GetBinLabel(i+1));
	  hHltEffSorted[Hltplacement[i]]->setBinContent(Hltbincounter[Hltplacement[i]]+1, hHlt_->GetBinContent(i+1));
	  hHltEffSorted[Hltplacement[i]]->setBinError(Hltbincounter[Hltplacement[i]]+1, hHlt_->GetBinError(i+1));
	  ++Hltbincounter[Hltplacement[i]];
	}
    }

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

  

