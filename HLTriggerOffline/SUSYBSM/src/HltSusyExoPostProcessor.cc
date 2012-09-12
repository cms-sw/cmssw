#include "HLTriggerOffline/SUSYBSM/interface/HltSusyExoPostProcessor.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include <iostream>
#include <string.h>
#include <iomanip>
#include <fstream>
#include <math.h>
#include "TH1.h"
#include "TProfile.h"

using namespace std;
using namespace edm;

HltSusyExoPostProcessor::HltSusyExoPostProcessor(const edm::ParameterSet& pset):
  subDir_(pset.getUntrackedParameter<std::string>("subDir", std::string("HLT/SusyExo"))),
  mcFlag(pset.getUntrackedParameter<bool>("mc_flag",true)),
  reco_parametersets(pset.getParameter<VParameterSet>("reco_parametersets")),
  mc_parametersets(pset.getParameter<VParameterSet>("mc_parametersets"))
{
  for(unsigned int i=0; i<reco_parametersets.size(); ++i)
    reco_dirs.push_back(reco_parametersets[i].getParameter<string>("name"));
  for(unsigned int i=0; i<mc_parametersets.size(); ++i)
    mc_dirs.push_back(mc_parametersets[i].getParameter<string>("name"));
}


void HltSusyExoPostProcessor::endRun(edm::Run const& run, edm::EventSetup const& es)
{

  LogDebug("HltSusyExoPostProcessor") << "Start endRun";
  //////////////////////////////////
  // setup DQM store              //
  //////////////////////////////////
  
  dqm = 0;
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
  std::string byEventDir = "/By_Event";
  std::string byMuonDir = "/By_Muon";



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
  double value = 0;
  double error = 0;

  //Calculate the efficiencies for histos without any selection
  dqm->setCurrentFolder(subDir_ + triggerBitsDir);
  //book the MonitorElements for the efficiencies 
//   MonitorElement* hL1EffBeforeCuts  = dqm->book1D("L1Eff", dqm->get(subDir_ + triggerBitsDir + "/L1Paths")->getTH1F());    
//   hL1EffBeforeCuts->setTitle("L1 Efficiencies");
//   MonitorElement* hHltEffBeforeCuts = dqm->book1D("HltEff",dqm->get(subDir_ + triggerBitsDir + "/HltPaths")->getTH1F());
//   hHltEffBeforeCuts->setTitle("HLT Efficiencies");

   TH1F* hL1Paths = dqm->get(subDir_ + triggerBitsDir + "/L1Paths")->getTH1F();
   MonitorElement* hL1EffBeforeCuts  = bookEffMEProfileFromTH1(hL1Paths, "Eff");
   TH1F* hHltPaths = dqm->get(subDir_ + triggerBitsDir + "/HltPaths")->getTH1F();
   MonitorElement* hHltEffBeforeCuts  = bookEffMEProfileFromTH1(hHltPaths, "Eff");



  LogDebug("HltSusyExoPostProcessor") << "MonitorElements booked";

  //get the total number of events 
  float nTotalBeforeCuts = -1;
  int reference_bin = -1;
  if(mcFlag)
    {
      nTotalBeforeCuts = dqm->get(subDir_ + triggerBitsDir + "/L1Bits")->getBinContent(nL1bins);
      reference_bin = nL1bins;
    }  
  else
    {
      for(int i=0; i<nHltbins; ++i)
	{
	  string triggername = ((TProfile*)hHltEffBeforeCuts->getTProfile())->GetXaxis()->GetBinLabel(i+1);
	  string eff_search = "HLT_ZeroBias";
	  if(triggername.find(eff_search)!=string::npos && triggername.size()==eff_search.size())
	    {
	      nTotalBeforeCuts = ((TProfile*)hHltEffBeforeCuts->getTProfile())->GetBinContent(i+1);
	      reference_bin = i+1;
	      break;
	    }
	}
      if(nTotalBeforeCuts == -1)
	{
	  nTotalBeforeCuts = dqm->get(subDir_ + triggerBitsDir + "/L1Bits")->getBinContent(nL1bins);
	  reference_bin = nHltbins;
	}
    }
  LogDebug("HltSusyExoPostProcessor") << "Total number of events = " << nTotalBeforeCuts;

  //fill the eff histo
  for(int i=0; i<nL1bins-1*mcFlag; i++) {
    value = (double) dqm->get(subDir_ + triggerBitsDir + "/L1Paths")->getBinContent(i+1) / (double) nTotalBeforeCuts;
    error = sqrt(value*(1-value)/(double)nTotalBeforeCuts);
     hL1EffBeforeCuts->setBinContent(i+1,value);
     hL1EffBeforeCuts->setBinEntries(i+1,1);
     hL1EffBeforeCuts->setBinError(i+1,error);
     //if(i!=nL1bins)
     // {
     string triggername = ((TProfile*)hL1EffBeforeCuts->getTProfile())->GetXaxis()->GetBinLabel(i+1);
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
  
  for(unsigned int i=0; i<L1placement.size(); ++i) 
    if(L1placement[i]!=-1)
      ++L1bins[L1placement[i]];

  for(int i=0; i<nHltbins-1*mcFlag; i++) {
    value = (double)dqm->get(subDir_ + triggerBitsDir + "/HltPaths")->getBinContent(i+1) / (double)nTotalBeforeCuts;
    error = sqrt(value*(1-value)/(double)nTotalBeforeCuts);
    hHltEffBeforeCuts->setBinContent(i+1,value);
    hHltEffBeforeCuts->setBinEntries(i+1,1);
    hHltEffBeforeCuts->setBinError(i+1,error);
    // if(i!=nHltbins)
    //  {
    string triggername = ((TProfile*)hHltEffBeforeCuts->getTProfile())->GetXaxis()->GetBinLabel(i+1);
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
  
  for(unsigned int i=0; i<Hltplacement.size(); ++i) 
    if(Hltplacement[i]!=-1)
      ++Hltbins[Hltplacement[i]];




  LogDebug("HltSusyExoPostProcessor") << "MonitorElements filled";

  //Create the sorted histograms
  dqm->setCurrentFolder(subDir_ + triggerBitsDir);
  MonitorElement* hL1EffSorted[7] = {dqm->bookProfile("L1_Mu", new TProfile("L1_Mu","Efficiencies of L1 Muon Triggers",L1bins[0],0,L1bins[0])),
				     dqm->bookProfile("L1_EG", new TProfile("L1_EG","Efficiencies of L1 EG Triggers",L1bins[1],0,L1bins[1])),
				     dqm->bookProfile("L1_Jet", new TProfile("L1_Jet","Efficiencies of L1 Jet Triggers",L1bins[2],0,L1bins[2])),
				     dqm->bookProfile("L1_ETM_ETT_HTT", new TProfile("L1_ETM_ETT_HTT","Efficiencies of L1 ETM, ETT, and HTT Triggers",L1bins[3],0,L1bins[3])),
				     dqm->bookProfile("L1_TauJet", new TProfile("L1_TauJet","Efficiencies of L1 TauJet Triggers",L1bins[4],0,L1bins[4])),
				     dqm->bookProfile("L1_XTrigger", new TProfile("L1_XTrigger","Efficiencies of L1 Cross Triggers",L1bins[5],0,L1bins[5])),
				     dqm->bookProfile("L1_Overflow", new TProfile("L1_Overflow","Efficiencies of L1 Unsorted Triggers",L1bins[6],0,L1bins[6])) };

  MonitorElement* hHltEffSorted[8] = {dqm->bookProfile("Hlt_Mu", new TProfile("Hlt_Mu","Efficiencies of HL Muon Triggers",Hltbins[0],0,Hltbins[0])),
				     dqm->bookProfile("Hlt_Ele", new TProfile("Hlt_Ele","Efficiencies of HL Electron Triggers",Hltbins[1],0,Hltbins[1])),
				      dqm->bookProfile("Hlt_Jet", new TProfile("Hlt_Jet","Efficiencies of HL Jet Triggers",Hltbins[2],0,Hltbins[2],"s")),
				     dqm->bookProfile("Hlt_Photon", new TProfile("Hlt_Photon","Efficiencies of HL Photon Triggers",Hltbins[3],0,Hltbins[3])),
				     dqm->bookProfile("Hlt_MET_HT", new TProfile("Hlt_MET_HT","Efficiencies of HL MET and HT Triggers",Hltbins[4],0,Hltbins[4])),
				     dqm->bookProfile("Hlt_Tau_BTag", new TProfile("Hlt_Tau_Btag","Efficiencies of HL Tau and BTag Triggers",Hltbins[5],0,Hltbins[5])),
				     dqm->bookProfile("Hlt_XTrigger", new TProfile("Hlt_XTrigger","Efficiencies of HL Cross Triggers",Hltbins[6],0,Hltbins[6])),
				     dqm->bookProfile("Hlt_Overflow", new TProfile("Hlt_Overflow","Efficiencies of HL Unsorted Triggers",Hltbins[7],0,Hltbins[7])) };

  int L1bincounter[8]; for(unsigned int i=0; i<sizeof(L1bincounter)/sizeof(L1bincounter[0]); ++i) L1bincounter[i]=0;
  int Hltbincounter[8]; for(unsigned int i=0; i<sizeof(Hltbincounter)/sizeof(Hltbincounter[0]); ++i) Hltbincounter[i]=0;
  TProfile* hL1_ = (TProfile*)hL1EffBeforeCuts->getTProfile();
  TProfile* hHlt_ = (TProfile*)hHltEffBeforeCuts->getTProfile();
  //  for(int i = 0; i<hHlt_->GetXaxis()->GetNbins(); i++) cout << "hHLT_->GetBinError(" << i << ") = " << hHlt_->GetBinError(i+1) << endl;
  for(unsigned int i=0; i<L1placement.size(); ++i)
    {
      if(L1placement[i]!=-1)
	{
	  hL1EffSorted[L1placement[i]]->setBinLabel(L1bincounter[L1placement[i]]+1, hL1_->GetXaxis()->GetBinLabel(i+1));
	  hL1EffSorted[L1placement[i]]->setBinContent(L1bincounter[L1placement[i]]+1, hL1_->GetBinContent(i+1));
	  hL1EffSorted[L1placement[i]]->setBinEntries(L1bincounter[L1placement[i]]+1, 1);
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
	  hHltEffSorted[Hltplacement[i]]->setBinEntries(Hltbincounter[Hltplacement[i]]+1, 1);
	  hHltEffSorted[Hltplacement[i]]->setBinError(Hltbincounter[Hltplacement[i]]+1, hHlt_->GetBinError(i+1));
	  ++Hltbincounter[Hltplacement[i]];
	}
    }

  for(unsigned int i=0; i<mc_dirs.size(); ++i)
    {
      //Calculate the efficiencies for histos after MC selection
      dqm->setCurrentFolder(subDir_ + mcSelBitsDir + "/" + mc_dirs[i]);
      //book the MonitorElements for the efficiencies 
      char set_name_L1[256], set_name_Hlt[256];
      sprintf(set_name_L1, "L1Eff_%s", mc_dirs[i].c_str());
      sprintf(set_name_Hlt, "HltEff_%s", mc_dirs[i].c_str());
      //      MonitorElement* hL1EffAfterMcCuts  = dqm->book1D(set_name_L1, dqm->get(subDir_ + mcSelBitsDir + "/" + mc_dirs[i] + "/L1Paths_" + mc_dirs[i])->getTH1F());
      MonitorElement* hL1EffAfterMcCuts  = bookEffMEProfileFromTH1(dqm->get(subDir_ + mcSelBitsDir + "/" + mc_dirs[i] + "/L1Paths_" + mc_dirs[i])->getTH1F(), (std::string) set_name_L1);
      hL1EffAfterMcCuts->setTitle("L1 Efficiencies for " + mc_dirs[i] + " selection");
      //      MonitorElement* hHltEffAfterMcCuts = dqm->book1D(set_name_Hlt, dqm->get(subDir_ + mcSelBitsDir + "/" + mc_dirs[i] + "/HltPaths_" + mc_dirs[i])->getTH1F());
      MonitorElement* hHltEffAfterMcCuts = bookEffMEProfileFromTH1(dqm->get(subDir_ + mcSelBitsDir + "/" + mc_dirs[i] + "/HltPaths_" + mc_dirs[i])->getTH1F(), (std::string) set_name_Hlt);
      hHltEffAfterMcCuts->setTitle("HLT Efficiencies for " + mc_dirs[i] + " selection");
      
      LogDebug("HltSusyExoPostProcessor") << "MonitorElements for " << mc_dirs[i] << " selection booked";
      
      //get the total number of events 

      float nTotalAfterMcCuts;
      if(mcFlag)
	nTotalAfterMcCuts = dqm->get(subDir_ + mcSelBitsDir + "/" + mc_dirs[i] + "/L1Bits_" + mc_dirs[i])->getBinContent(nL1bins);
      else
	nTotalAfterMcCuts = dqm->get(subDir_ + mcSelBitsDir + "/" + mc_dirs[i] + "/HltBits_" + mc_dirs[i])->getBinContent(reference_bin);

      LogDebug("HltSusyExoPostProcessor") << "Total number of events = " << nTotalAfterMcCuts;

      MonitorElement* hL1EffSorted_mc[7];
      MonitorElement* hHltEffSorted_mc[8];
      char buffer1[256], buffer2[256];
      string L1_nametags[7] = {"Mu","EG","Jet","ETM_ETT_HTT","TauJet","XTrigger","Overflow"};
      string L1_titletags[7] = {"Muon","EG","Jet","ETM, ETT, and HTT","TauJet","Cross","Unsorted"};
      string Hlt_nametags[8] = {"Mu","Ele","Jet","Photon","MET_HT","Tau_BTag","XTrigger","Overflow"};
      string Hlt_titletags[8] = {"Muon","Electron","Jet","Photon","MET and HT","Tau and BTag","Cross","Unsorted"};
      for(unsigned int j=0; j<sizeof(hL1EffSorted_mc)/sizeof(hL1EffSorted_mc[0]); ++j)
	{
	  sprintf(buffer1,"L1_%s_%s",mc_dirs[i].c_str(),L1_nametags[j].c_str());
	  sprintf(buffer2,"Efficiencies of L1 %s Triggers for %s Selection",L1_titletags[j].c_str(),mc_dirs[i].c_str());
	  hL1EffSorted_mc[j] = dqm->bookProfile(buffer1, new TProfile(buffer1,buffer2,L1bins[j],0,L1bins[j]));
	}
      for(unsigned int j=0; j<sizeof(hHltEffSorted_mc)/sizeof(hHltEffSorted_mc[0]); ++j)
	{
	  sprintf(buffer1,"Hlt_%s_%s",mc_dirs[i].c_str(),Hlt_nametags[j].c_str());
	  sprintf(buffer2,"Efficiencies of HL %s Triggers for %s Selection",Hlt_titletags[j].c_str(),mc_dirs[i].c_str());
	  hHltEffSorted_mc[j] = dqm->bookProfile(buffer1, new TProfile(buffer1,buffer2,Hltbins[j],0,Hltbins[j]));
	}
      /*
      MonitorElement* hL1EffSorted_mc[7] = {dqm->book1D("L1_"+mc_dirs[i]+"_Mu", new TH1F("L1_"+mc_dirs[i].c_str()+"_Mu","Efficiencies of L1 Muon Triggers for "+mc_dirs[i]+" Selection",L1bins[0],0,L1bins[0])),
					    dqm->book1D("L1_"+mc_dirs[i]+"_EG", new TH1F("L1_"+mc_dirs[i]+"_EG","Efficiencies of L1 EG Triggers for "+mc_dirs[i]+" Selection",L1bins[1],0,L1bins[1])),
					    dqm->book1D("L1_"+mc_dirs[i]+"_Jet", new TH1F("L1_"+mc_dirs[i]+"_Jet","Efficiencies of L1 Jet Triggers for "+mc_dirs[i]+" Selection",L1bins[2],0,L1bins[2])),
					    dqm->book1D("L1_"+mc_dirs[i]+"_ETM_ETT_HTT", new TH1F("L1_"+mc_dirs[i]+"_ETM_ETT_HTT","Efficiencies of L1 ETM, ETT, and HTT Triggers for "+mc_dirs[i]+" Selection",L1bins[3],0,L1bins[3])),
					    dqm->book1D("L1_"+mc_dirs[i]+"_TauJet", new TH1F("L1_"+mc_dirs[i]+"_TauJet","Efficiencies of L1 TauJet Triggers for "+mc_dirs[i]+" Selection",L1bins[4],0,L1bins[4])),
					    dqm->book1D("L1_"+mc_dirs[i]+"_XTrigger", new TH1F("L1_"+mc_dirs[i]+"_XTrigger","Efficiencies of L1 Cross Triggers for "+mc_dirs[i]+" Selection",L1bins[5],0,L1bins[5])),
					    dqm->book1D("L1_"+mc_dirs[i]+"_Overflow", new TH1F("L1_"+mc_dirs[i]+"_Overflow","Efficiencies of L1 Unsorted Triggers for "+mc_dirs[i]+" Selection",L1bins[6],0,L1bins[6])) };
      MonitorElement* hHltEffSorted_mc[8] = {dqm->book1D("Hlt_"+mc_dirs[i]+"_Mu", new TH1F("Hlt_"+mc_dirs[i]+"_Mu","Efficiencies of HL Muon Triggers for "+mc_dirs[i]+" Selection",Hltbins[0],0,Hltbins[0])),
					     dqm->book1D("Hlt_"+mc_dirs[i]+"_Ele", new TH1F("Hlt_"+mc_dirs[i]+"_Ele","Efficiencies of HL Electron Triggers for "+mc_dirs[i]+" Selection",Hltbins[1],0,Hltbins[1])),
					     dqm->book1D("Hlt_"+mc_dirs[i]+"_Jet", new TH1F("Hlt_"+mc_dirs[i]+"_Jet","Efficiencies of HL Jet Triggers for "+mc_dirs[i]+" Selection",Hltbins[2],0,Hltbins[2])),
					     dqm->book1D("Hlt_"+mc_dirs[i]+"_Photon", new TH1F("Hlt_"+mc_dirs[i]+"_Photon","Efficiencies of HL Photon Triggers for "+mc_dirs[i]+" Selection",Hltbins[3],0,Hltbins[3])),
					     dqm->book1D("Hlt_"+mc_dirs[i]+"_MET_HT", new TH1F("Hlt_"+mc_dirs[i]+"_MET_HT","Efficiencies of HL MET and HT Triggers for "+mc_dirs[i]+" Selection",Hltbins[4],0,Hltbins[4])),
					     dqm->book1D("Hlt_"+mc_dirs[i]+"_Tau_BTag", new TH1F("Hlt_"+mc_dirs[i]+"_Tau_Btag","Efficiencies of HL Tau and BTag Triggers for "+mc_dirs[i]+" Selection",Hltbins[5],0,Hltbins[5])),
					     dqm->book1D("Hlt_"+mc_dirs[i]+"_XTrigger", new TH1F("Hlt_"+mc_dirs[i]+"_XTrigger","Efficiencies of HL Cross Triggers for "+mc_dirs[i]+" Selection",Hltbins[6],0,Hltbins[6])),
					     dqm->book1D("Hlt_"+mc_dirs[i]+"_Overflow", new TH1F("Hlt_"+mc_dirs[i]+"_Overflow","Efficiencies of HL Unsorted Triggers for "+mc_dirs[i]+" Selection",Hltbins[7],0,Hltbins[7])) };*/

      //fill the eff histo
      int L1bincounter_mc[8]; for(unsigned int j=0; j<sizeof(L1bincounter_mc)/sizeof(L1bincounter_mc[0]); ++j) L1bincounter_mc[j]=0;
      int Hltbincounter_mc[8]; for(unsigned int j=0; j<sizeof(Hltbincounter_mc)/sizeof(Hltbincounter_mc[0]); ++j) Hltbincounter_mc[j]=0;
      TProfile* hL1_mc = (TProfile*)hL1EffAfterMcCuts->getTProfile();
      TProfile* hHlt_mc = (TProfile*)hHltEffAfterMcCuts->getTProfile();
      for(unsigned int j=0; j<L1placement.size(); j++) {
	value = nTotalAfterMcCuts ? (float)dqm->get(subDir_ + mcSelBitsDir + "/" + mc_dirs[i] + "/L1Paths_" + mc_dirs[i])->getBinContent(j+1) / nTotalAfterMcCuts : 0;
	error = nTotalAfterMcCuts ? sqrt(value*(1-value)/nTotalAfterMcCuts) : 0;
	hL1EffAfterMcCuts->setBinContent(j+1,value);
	hL1EffAfterMcCuts->setBinEntries(j+1,1);
	hL1EffAfterMcCuts->setBinError(j+1,error);
	if(L1placement[j]!=-1)
	  {
	    hL1EffSorted_mc[L1placement[j]]->setBinLabel(L1bincounter_mc[L1placement[j]]+1, hL1_mc->GetXaxis()->GetBinLabel(j+1));
	    hL1EffSorted_mc[L1placement[j]]->setBinContent(L1bincounter_mc[L1placement[j]]+1, hL1_mc->GetBinContent(j+1));
	    hL1EffSorted_mc[L1placement[j]]->setBinEntries(L1bincounter_mc[L1placement[j]]+1, 1);
	    hL1EffSorted_mc[L1placement[j]]->setBinError(L1bincounter_mc[L1placement[j]]+1, hL1_mc->GetBinError(j+1));
	    ++L1bincounter_mc[L1placement[j]];
	  }
      }
      if(nL1bins!=int(L1placement.size()))
	{
	  value = nTotalAfterMcCuts ? (float)dqm->get(subDir_ + mcSelBitsDir + "/" + mc_dirs[i] + "/L1Paths_" + mc_dirs[i])->getBinContent(nL1bins) / nTotalAfterMcCuts : 0;
	  error = nTotalAfterMcCuts ? sqrt(value*(1-value)/nTotalAfterMcCuts) : 0;
	  hL1EffAfterMcCuts->setBinContent(nL1bins,value);
	  hL1EffAfterMcCuts->setBinEntries(nL1bins,1);
	  hL1EffAfterMcCuts->setBinError(nL1bins,error);
	}
      for(unsigned int j=0; j<Hltplacement.size(); j++) {
	value = nTotalAfterMcCuts ? (float)dqm->get(subDir_ + mcSelBitsDir + "/" + mc_dirs[i] + "/HltPaths_" + mc_dirs[i])->getBinContent(j+1) / nTotalAfterMcCuts : 0;
	error = nTotalAfterMcCuts ? sqrt(value*(1-value)/nTotalAfterMcCuts) : 0;
	hHltEffAfterMcCuts->setBinContent(j+1,value);
	hHltEffAfterMcCuts->setBinEntries(j+1,1);
	hHltEffAfterMcCuts->setBinError(j+1,error);
	if(Hltplacement[j]!=-1)
	  {
	    hHltEffSorted_mc[Hltplacement[j]]->setBinLabel(Hltbincounter_mc[Hltplacement[j]]+1, hHlt_mc->GetXaxis()->GetBinLabel(j+1));
	    hHltEffSorted_mc[Hltplacement[j]]->setBinContent(Hltbincounter_mc[Hltplacement[j]]+1, hHlt_mc->GetBinContent(j+1));
	    hHltEffSorted_mc[Hltplacement[j]]->setBinEntries(Hltbincounter_mc[Hltplacement[j]]+1, 1);
	    hHltEffSorted_mc[Hltplacement[j]]->setBinError(Hltbincounter_mc[Hltplacement[j]]+1, hHlt_mc->GetBinError(j+1));
	    ++Hltbincounter_mc[Hltplacement[j]];
	  }
      }
      if(nHltbins!=int(Hltplacement.size()))
	{
	  value = nTotalAfterMcCuts ? (float)dqm->get(subDir_ + mcSelBitsDir + "/" + mc_dirs[i] + "/HltPaths_" + mc_dirs[i])->getBinContent(nHltbins) / nTotalAfterMcCuts : 0;
	  error = nTotalAfterMcCuts ? sqrt(value*(1-value)/nTotalAfterMcCuts) : 0;
	  hHltEffAfterMcCuts->setBinContent(nHltbins,value);
	  hHltEffAfterMcCuts->setBinEntries(nHltbins,1);
	  hHltEffAfterMcCuts->setBinError(nHltbins,error);
	}
      LogDebug("HltSusyExoPostProcessor") << "MonitorElements filled for " << mc_dirs[i] <<" selection";
    }

  for(unsigned int i=0; i<reco_dirs.size(); ++i)
    {
      //Calculate the efficiencies for histos after RECO selection
      dqm->setCurrentFolder(subDir_ + recoSelBitsDir + "/" + reco_dirs[i]);
      //book the MonitorElements for the efficiencies 
      char set_name_L1[256], set_name_Hlt[256];
      sprintf(set_name_L1, "L1Eff_%s", reco_dirs[i].c_str());
      sprintf(set_name_Hlt, "HltEff_%s", reco_dirs[i].c_str());
      //      MonitorElement* hL1EffAfterRecoCuts  = dqm->book1D(set_name_L1, dqm->get(subDir_ + recoSelBitsDir + "/" + reco_dirs[i] + "/L1Paths_" + reco_dirs[i])->getTH1F());    
      MonitorElement* hL1EffAfterRecoCuts  = bookEffMEProfileFromTH1(dqm->get(subDir_ + recoSelBitsDir + "/" + reco_dirs[i] + "/L1Paths_" + reco_dirs[i])->getTH1F(), (std::string) set_name_L1);
      hL1EffAfterRecoCuts->setTitle("L1 Efficiencies for " + reco_dirs[i] + " selection");
      //      MonitorElement* hHltEffAfterRecoCuts = dqm->book1D(set_name_Hlt, dqm->get(subDir_ + recoSelBitsDir + "/" + reco_dirs[i] + "/HltPaths_" + reco_dirs[i])->getTH1F());
      MonitorElement* hHltEffAfterRecoCuts = bookEffMEProfileFromTH1(dqm->get(subDir_ + recoSelBitsDir + "/" + reco_dirs[i] + "/HltPaths_" + reco_dirs[i])->getTH1F(), (std::string) set_name_Hlt);
      hHltEffAfterRecoCuts->setTitle("HLT Efficiencies for " + reco_dirs[i] + " selection");

      LogDebug("HltSusyExoPostProcessor") << "MonitorElements for " << reco_dirs[i] << " selection booked";
      
      //get the total number of events 
      float nTotalAfterRecoCuts;
      if(mcFlag)
	nTotalAfterRecoCuts = dqm->get(subDir_ + recoSelBitsDir + "/" + reco_dirs[i] + "/L1Bits_" + reco_dirs[i])->getBinContent(nL1bins);
      else
	nTotalAfterRecoCuts = dqm->get(subDir_ + recoSelBitsDir + "/" + reco_dirs[i] + "/HltBits_" + reco_dirs[i])->getBinContent(reference_bin);

      LogDebug("HltSusyExoPostProcessor") << "Total number of events = " << nTotalAfterRecoCuts;
      
      MonitorElement* hL1EffSorted_reco[7];
      MonitorElement* hHltEffSorted_reco[8];
      char buffer1[256], buffer2[256];
      string L1_nametags[7] = {"Mu","EG","Jet","ETM_ETT_HTT","TauJet","XTrigger","Overflow"};
      string L1_titletags[7] = {"Muon","EG","Jet","ETM, ETT, and HTT","TauJet","Cross","Unsorted"};
      string Hlt_nametags[8] = {"Mu","Ele","Jet","Photon","MET_HT","Tau_BTag","XTrigger","Overflow"};
      string Hlt_titletags[8] = {"Muon","Electron","Jet","Photon","MET and HT","Tau and BTag","Cross","Unsorted"};
      for(unsigned int j=0; j<sizeof(hL1EffSorted_reco)/sizeof(hL1EffSorted_reco[0]); ++j)
	{
	  sprintf(buffer1,"L1_%s_%s",reco_dirs[i].c_str(),L1_nametags[j].c_str());
	  sprintf(buffer2,"Efficiencies of L1 %s Triggers for %s Selection",L1_titletags[j].c_str(),reco_dirs[i].c_str()); 
	  hL1EffSorted_reco[j] = dqm->bookProfile(buffer1, new TProfile(buffer1,buffer2,L1bins[j],0,L1bins[j]));
	}
      for(unsigned int j=0; j<sizeof(hHltEffSorted_reco)/sizeof(hHltEffSorted_reco[0]); ++j)
	{
	  sprintf(buffer1,"Hlt_%s_%s",reco_dirs[i].c_str(),Hlt_nametags[j].c_str());
	  sprintf(buffer2,"Efficiencies of HL %s Triggers for %s Selection",Hlt_titletags[j].c_str(),reco_dirs[i].c_str()); 
	  hHltEffSorted_reco[j] = dqm->bookProfile(buffer1, new TProfile(buffer1,buffer2,Hltbins[j],0,Hltbins[j]));
	}
      /*
      MonitorElement* hL1EffSorted_reco[7] = {dqm->book1D("L1_"+reco_dirs[i]+"_Mu", new TH1F("L1_"+reco_dirs[i]+"_Mu","Efficiencies of L1 Muon Triggers for "+reco_dirs[i]+" Selection",L1bins[0],0,L1bins[0])),
					      dqm->book1D("L1_"+reco_dirs[i]+"_EG", new TH1F("L1_"+reco_dirs[i]+"_EG","Efficiencies of L1 EG Triggers for "+reco_dirs[i]+" Selection",L1bins[1],0,L1bins[1])),
					      dqm->book1D("L1_"+reco_dirs[i]+"_Jet", new TH1F("L1_"+reco_dirs[i]+"_Jet","Efficiencies of L1 Jet Triggers for "+reco_dirs[i]+" Selection",L1bins[2],0,L1bins[2])),
					      dqm->book1D("L1_"+reco_dirs[i]+"_ETM_ETT_HTT", new TH1F("L1_"+reco_dirs[i]+"_ETM_ETT_HTT","Efficiencies of L1 ETM, ETT, and HTT Triggers for "+reco_dirs[i]+" Selection",L1bins[3],0,L1bins[3])),
					      dqm->book1D("L1_"+reco_dirs[i]+"_TauJet", new TH1F("L1_"+reco_dirs[i]+"_TauJet","Efficiencies of L1 TauJet Triggers for "+reco_dirs[i]+" Selection",L1bins[4],0,L1bins[4])),
					      dqm->book1D("L1_"+reco_dirs[i]+"_XTrigger", new TH1F("L1_"+reco_dirs[i]+"_XTrigger","Efficiencies of L1 Cross Triggers for "+reco_dirs[i]+" Selection",L1bins[5],0,L1bins[5])),
					      dqm->book1D("L1_"+reco_dirs[i]+"_Overflow", new TH1F("L1_"+reco_dirs[i]+"_Overflow","Efficiencies of L1 Unsorted Triggers for "+reco_dirs[i]+" Selection",L1bins[6],0,L1bins[6])) };
      MonitorElement* hHltEffSorted_reco[8] = {dqm->book1D("Hlt_"+reco_dirs[i]+"_Mu", new TH1F("Hlt_"+reco_dirs[i]+"_Mu","Efficiencies of HL Muon Triggers for "+reco_dirs[i]+" Selection",Hltbins[0],0,Hltbins[0])),
					       dqm->book1D("Hlt_"+reco_dirs[i]+"_Ele", new TH1F("Hlt_"+reco_dirs[i]+"_Ele","Efficiencies of HL Electron Triggers for "+reco_dirs[i]+" Selection",Hltbins[1],0,Hltbins[1])),
					       dqm->book1D("Hlt_"+reco_dirs[i]+"_Jet", new TH1F("Hlt_"+reco_dirs[i]+"_Jet","Efficiencies of HL Jet Triggers for "+reco_dirs[i]+" Selection",Hltbins[2],0,Hltbins[2])),
					       dqm->book1D("Hlt_"+reco_dirs[i]+"_Photon", new TH1F("Hlt_"+reco_dirs[i]+"_Photon","Efficiencies of HL Photon Triggers for "+reco_dirs[i]+" Selection",Hltbins[3],0,Hltbins[3])),
					       dqm->book1D("Hlt_"+reco_dirs[i]+"_MET_HT", new TH1F("Hlt_"+reco_dirs[i]+"_MET_HT","Efficiencies of HL MET and HT Triggers for "+reco_dirs[i]+" Selection",Hltbins[4],0,Hltbins[4])),
					       dqm->book1D("Hlt_"+reco_dirs[i]+"_Tau_BTag", new TH1F("Hlt_"+reco_dirs[i]+"_Tau_Btag","Efficiencies of HL Tau and BTag Triggers for "+reco_dirs[i]+" Selection",Hltbins[5],0,Hltbins[5])),
					       dqm->book1D("Hlt_"+reco_dirs[i]+"_XTrigger", new TH1F("Hlt_"+reco_dirs[i]+"_XTrigger","Efficiencies of HL Cross Triggers for "+reco_dirs[i]+" Selection",Hltbins[6],0,Hltbins[6])),
					       dqm->book1D("Hlt_"+reco_dirs[i]+"_Overflow", new TH1F("Hlt_"+reco_dirs[i]+"_Overflow","Efficiencies of HL Unsorted Triggers for "+reco_dirs[i]+" Selection",Hltbins[7],0,Hltbins[7])) };
      */

      //fill the eff histo
      int L1bincounter_reco[8]; for(unsigned int j=0; j<sizeof(L1bincounter_reco)/sizeof(L1bincounter_reco[0]); ++j) L1bincounter_reco[j]=0;
      int Hltbincounter_reco[8]; for(unsigned int j=0; j<sizeof(Hltbincounter_reco)/sizeof(Hltbincounter_reco[0]); ++j) Hltbincounter_reco[j]=0;
      TProfile* hL1_reco = (TProfile*)hL1EffAfterRecoCuts->getTProfile();
      TProfile* hHlt_reco = (TProfile*)hHltEffAfterRecoCuts->getTProfile();
      for(unsigned int j=0; j<L1placement.size(); j++) {
	value = nTotalAfterRecoCuts ? (float)dqm->get(subDir_ + recoSelBitsDir + "/" + reco_dirs[i] + "/L1Paths_" + reco_dirs[i])->getBinContent(j+1) / nTotalAfterRecoCuts : 0;
	error = nTotalAfterRecoCuts ? sqrt(value*(1-value)/nTotalAfterRecoCuts) : 0;
	hL1EffAfterRecoCuts->setBinContent(j+1,value);
	hL1EffAfterRecoCuts->setBinEntries(j+1,1);
	hL1EffAfterRecoCuts->setBinError(j+1,error);
	if(L1placement[j]!=-1)
	  {
	    hL1EffSorted_reco[L1placement[j]]->setBinLabel(L1bincounter_reco[L1placement[j]]+1, hL1_reco->GetXaxis()->GetBinLabel(j+1));
	    hL1EffSorted_reco[L1placement[j]]->setBinContent(L1bincounter_reco[L1placement[j]]+1, hL1_reco->GetBinContent(j+1));
	    hL1EffSorted_reco[L1placement[j]]->setBinEntries(L1bincounter_reco[L1placement[j]]+1, 1);
	    hL1EffSorted_reco[L1placement[j]]->setBinError(L1bincounter_reco[L1placement[j]]+1, hL1_reco->GetBinError(j+1));
	    ++L1bincounter_reco[L1placement[j]];
	  }
      }
      if(nL1bins!=int(L1placement.size()))
	{
	  value = nTotalAfterRecoCuts ? (float)dqm->get(subDir_ + recoSelBitsDir + "/" + reco_dirs[i] + "/L1Paths_" + reco_dirs[i])->getBinContent(nL1bins) / nTotalAfterRecoCuts : 0;
	  error = nTotalAfterRecoCuts ? sqrt(value*(1-value)/nTotalAfterRecoCuts) : 0;
	  hL1EffAfterRecoCuts->setBinContent(nL1bins,value);
	  hL1EffAfterRecoCuts->setBinEntries(nL1bins,1);
	  hL1EffAfterRecoCuts->setBinError(nL1bins,error);
	}
      for(unsigned int j=0; j<Hltplacement.size(); j++) {
	value = nTotalAfterRecoCuts ? (float)dqm->get(subDir_ + recoSelBitsDir + "/" + reco_dirs[i] + "/HltPaths_" + reco_dirs[i])->getBinContent(j+1) / nTotalAfterRecoCuts : 0;
	error = nTotalAfterRecoCuts ? sqrt(value*(1-value)/nTotalAfterRecoCuts) : 0;
	hHltEffAfterRecoCuts->setBinContent(j+1,value);
	hHltEffAfterRecoCuts->setBinEntries(j+1,1);
	hHltEffAfterRecoCuts->setBinError(j+1,error);
	if(Hltplacement[j]!=-1)
	  {
	    hHltEffSorted_reco[Hltplacement[j]]->setBinLabel(Hltbincounter_reco[Hltplacement[j]]+1, hHlt_reco->GetXaxis()->GetBinLabel(j+1));
	    hHltEffSorted_reco[Hltplacement[j]]->setBinContent(Hltbincounter_reco[Hltplacement[j]]+1, hHlt_reco->GetBinContent(j+1));
	    hHltEffSorted_reco[Hltplacement[j]]->setBinEntries(Hltbincounter_reco[Hltplacement[j]]+1, 1);
	    hHltEffSorted_reco[Hltplacement[j]]->setBinError(Hltbincounter_reco[Hltplacement[j]]+1, hHlt_reco->GetBinError(j+1));
	    ++Hltbincounter_reco[Hltplacement[j]];
	  }
      }
      if(nHltbins!=int(Hltplacement.size()))
	{
	  value = nTotalAfterRecoCuts ? (float)dqm->get(subDir_ + recoSelBitsDir + "/" + reco_dirs[i] + "/HltPaths_" + reco_dirs[i])->getBinContent(nHltbins) / nTotalAfterRecoCuts : 0;
	  error = nTotalAfterRecoCuts ? sqrt(value*(1-value)/nTotalAfterRecoCuts) : 0;
	  hHltEffAfterRecoCuts->setBinContent(nHltbins,value);
	  hHltEffAfterRecoCuts->setBinEntries(nHltbins,1);
	  hHltEffAfterRecoCuts->setBinError(nHltbins,error);
	}
      LogDebug("HltSusyExoPostProcessor") << "MonitorElements filled";
    }

  int pt_bins=100, eta_bins=100;
  double pt_floor=0., pt_ceiling = 200., eta_floor=-3.5, eta_ceiling=3.5;
  dqm->setCurrentFolder(subDir_ + byEventDir);
  MonitorElement* hPt_1_ByEvent = dqm->book1D("Pt_1_ByEvent","Turn on as a Function of P_{t}, |Eta|<1.2, By Event",pt_bins, pt_floor, pt_ceiling);
  MonitorElement* hEta_1_ByEvent= dqm->book1D("Eta_1_ByEvent","Efficiency as a Function of Eta, P_{t}>0, By Event", eta_bins, eta_floor,eta_ceiling);
  MonitorElement* hPt_2_ByEvent = dqm->book1D("Pt_2_ByEvent","Turn on as a Function of P_{t}, 1.2<|Eta|<2.1, By Event",pt_bins, pt_floor, pt_ceiling);
  MonitorElement* hEta_2_ByEvent= dqm->book1D("Eta_2_ByEvent","Efficiency as a Function of Eta, P_{t}>10, By Event", eta_bins, eta_floor,eta_ceiling);
  MonitorElement* hPt_3_ByEvent = dqm->book1D("Pt_3_ByEvent","Turn on as a Function of P_{t}, |Eta|>2.1, By Event",pt_bins, pt_floor, pt_ceiling);
  MonitorElement* hEta_3_ByEvent= dqm->book1D("Eta_3_ByEvent","Efficiency as a Function of Eta, P_{t}>20, By Event", eta_bins, eta_floor,eta_ceiling);
  dqm->setCurrentFolder(subDir_ + byMuonDir);
  MonitorElement* hPt_1_ByMuon = dqm->book1D("Pt_1_ByMuon","Turn on as a Function of P_{t}, |Eta|<1.2, By Muon",pt_bins, pt_floor, pt_ceiling);
  MonitorElement* hEta_1_ByMuon= dqm->book1D("Eta_1_ByMuon","Efficiency as a Function of Eta, P_{t}>0, By Muon", eta_bins, eta_floor,eta_ceiling);
  MonitorElement* hPt_2_ByMuon = dqm->book1D("Pt_2_ByMuon","Turn on as a Function of P_{t}, 1.2<|Eta|<2.1, By Muon",pt_bins, pt_floor, pt_ceiling);
  MonitorElement* hEta_2_ByMuon= dqm->book1D("Eta_2_ByMuon","Efficiency as a Function of Eta, P_{t}>10, By Muon", eta_bins, eta_floor,eta_ceiling);
  MonitorElement* hPt_3_ByMuon = dqm->book1D("Pt_3_ByMuon","Turn on as a Function of P_{t}, |Eta|>2.1, By Muon",pt_bins, pt_floor, pt_ceiling);
  MonitorElement* hEta_3_ByMuon= dqm->book1D("Eta_3_ByMuon","Efficiency as a Function of Eta, P_{t}>20, By Muon", eta_bins, eta_floor,eta_ceiling);

  for(int i=1; i<=hPt_1_ByEvent->getNbinsX(); ++i)
    {
      double n1 = (double)dqm->get(subDir_ + byEventDir + "/LeadAssocRecoMuonPt_1_ByEvent")->getBinContent(i);
      double n2 = (double)dqm->get(subDir_ + byEventDir + "/LeadRecoMuonPt_1_ByEvent")->getBinContent(i);
      double value = (n2!=0 ? n1/n2 : 0);
      double error = (n2!=0 ? sqrt(value*(1-value)/n2) : 0);
      hPt_1_ByEvent->setBinContent(i,value);
      hPt_1_ByEvent->setBinError(i,error);

      n1 = (double)dqm->get(subDir_ + byEventDir + "/LeadAssocRecoMuonPt_2_ByEvent")->getBinContent(i);
      n2 = (double)dqm->get(subDir_ + byEventDir + "/LeadRecoMuonPt_2_ByEvent")->getBinContent(i);
      value = (n2!=0 ? n1/n2 : 0);
      error = (n2!=0 ? sqrt(value*(1-value)/n2) : 0);
      hPt_2_ByEvent->setBinContent(i,value);
      hPt_2_ByEvent->setBinError(i,error);

      n1 = (double)dqm->get(subDir_ + byEventDir + "/LeadAssocRecoMuonPt_3_ByEvent")->getBinContent(i);
      n2 = (double)dqm->get(subDir_ + byEventDir + "/LeadRecoMuonPt_3_ByEvent")->getBinContent(i);
      value = (n2!=0 ? n1/n2 : 0);
      error = (n2!=0 ? sqrt(value*(1-value)/n2) : 0);
      hPt_3_ByEvent->setBinContent(i,value);
      hPt_3_ByEvent->setBinError(i,error);

      n1 = (double)dqm->get(subDir_ + byMuonDir + "/AssocRecoMuonPt_1_ByMuon")->getBinContent(i);
      n2 = (double)dqm->get(subDir_ + byMuonDir + "/RecoMuonPt_1_ByMuon")->getBinContent(i);
      value = (n2!=0 ? n1/n2 : 0);
      error = (n2!=0 ? sqrt(value*(1-value)/n2) : 0);
      hPt_1_ByMuon->setBinContent(i,value);
      hPt_1_ByMuon->setBinError(i,error);

      n1 = (double)dqm->get(subDir_ + byMuonDir + "/AssocRecoMuonPt_2_ByMuon")->getBinContent(i);
      n2 = (double)dqm->get(subDir_ + byMuonDir + "/RecoMuonPt_2_ByMuon")->getBinContent(i);
      value = (n2!=0 ? n1/n2 : 0);
      error = (n2!=0 ? sqrt(value*(1-value)/n2) : 0);
      hPt_2_ByMuon->setBinContent(i,value);
      hPt_2_ByMuon->setBinError(i,error);

      n1 = (double)dqm->get(subDir_ + byMuonDir + "/AssocRecoMuonPt_3_ByMuon")->getBinContent(i);
      n2 = (double)dqm->get(subDir_ + byMuonDir + "/RecoMuonPt_3_ByMuon")->getBinContent(i);
      value = (n2!=0 ? n1/n2 : 0);
      error = (n2!=0 ? sqrt(value*(1-value)/n2) : 0);
      hPt_3_ByMuon->setBinContent(i,value);
      hPt_3_ByMuon->setBinError(i,error);
    }

  for(int i=1; i<=hEta_1_ByEvent->getNbinsX(); ++i)
    {
      double n1 = (double)dqm->get(subDir_ + byEventDir + "/LeadAssocRecoMuonEta_1_ByEvent")->getBinContent(i);
      double n2 = (double)dqm->get(subDir_ + byEventDir + "/LeadRecoMuonEta_1_ByEvent")->getBinContent(i);
      double value = (n2!=0 ? n1/n2 : 0);
      double error = (n2!=0 ? sqrt(value*(1-value)/n2) : 0);
      hEta_1_ByEvent->setBinContent(i,value);
      hEta_1_ByEvent->setBinError(i,error);

      n1 = (double)dqm->get(subDir_ + byEventDir + "/LeadAssocRecoMuonEta_2_ByEvent")->getBinContent(i);
      n2 = (double)dqm->get(subDir_ + byEventDir + "/LeadRecoMuonEta_2_ByEvent")->getBinContent(i);
      value = (n2!=0 ? n1/n2 : 0);
      error = (n2!=0 ? sqrt(value*(1-value)/n2) : 0);
      hEta_2_ByEvent->setBinContent(i,value);
      hEta_2_ByEvent->setBinError(i,error);

      n1 = (double)dqm->get(subDir_ + byEventDir + "/LeadAssocRecoMuonEta_3_ByEvent")->getBinContent(i);
      n2 = (double)dqm->get(subDir_ + byEventDir + "/LeadRecoMuonEta_3_ByEvent")->getBinContent(i);
      value = (n2!=0 ? n1/n2 : 0);
      error = (n2!=0 ? sqrt(value*(1-value)/n2) : 0);
      hEta_3_ByEvent->setBinContent(i,value);
      hEta_3_ByEvent->setBinError(i,error);

      n1 = (double)dqm->get(subDir_ + byMuonDir + "/AssocRecoMuonEta_1_ByMuon")->getBinContent(i);
      n2 = (double)dqm->get(subDir_ + byMuonDir + "/RecoMuonEta_1_ByMuon")->getBinContent(i);
      value = (n2!=0 ? n1/n2 : 0);
      error = (n2!=0 ? sqrt(value*(1-value)/n2) : 0);
      hEta_1_ByMuon->setBinContent(i,value);
      hEta_1_ByMuon->setBinError(i,error);

      n1 = (double)dqm->get(subDir_ + byMuonDir + "/AssocRecoMuonEta_2_ByMuon")->getBinContent(i);
      n2 = (double)dqm->get(subDir_ + byMuonDir + "/RecoMuonEta_2_ByMuon")->getBinContent(i);
      value = (n2!=0 ? n1/n2 : 0);
      error = (n2!=0 ? sqrt(value*(1-value)/n2) : 0);
      hEta_2_ByMuon->setBinContent(i,value);
      hEta_2_ByMuon->setBinError(i,error);

      n1 = (double)dqm->get(subDir_ + byMuonDir + "/AssocRecoMuonEta_3_ByMuon")->getBinContent(i);
      n2 = (double)dqm->get(subDir_ + byMuonDir + "/RecoMuonEta_3_ByMuon")->getBinContent(i);
      value = (n2!=0 ? n1/n2 : 0);
      error = (n2!=0 ? sqrt(value*(1-value)/n2) : 0);
      hEta_3_ByMuon->setBinContent(i,value);
      hEta_3_ByMuon->setBinError(i,error);
    }
}




MonitorElement* HltSusyExoPostProcessor::bookEffMEProfileFromTH1(TH1F* histo, std::string name) {
  MonitorElement* myEffME;
  if(name == "Eff") {
    //    myEffME = dqm->bookProfile("Eff_"+((std::string) histo->GetName()), new TProfile(name.c_str(), histo->GetTitle(), histo->GetXaxis()->GetNbins(), histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax(),"s"));
    myEffME = dqm->bookProfile((std::string) ("Eff_"+((std::string) histo->GetName())), ((std::string) histo->GetTitle()), histo->GetXaxis()->GetNbins(), histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax(), 100, 0, 1, "");
  }
  else {
    myEffME = dqm->bookProfile(name, new TProfile(name.c_str(), histo->GetTitle(), histo->GetXaxis()->GetNbins(), histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax()));
  }
  for(int i=0; i<histo->GetXaxis()->GetNbins(); i++) {
    myEffME->setBinLabel(i+1, histo->GetXaxis()->GetBinLabel(i+1),1);
  }
  return myEffME;
}

  


DEFINE_FWK_MODULE(HltSusyExoPostProcessor);

  

