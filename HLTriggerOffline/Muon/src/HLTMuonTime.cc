/** \class HLTMuonTimeTime
 *  Get L1/HLT efficiency/rate plots
 *
 *  \author M. Vander Donckt  
 */

#include "HLTriggerOffline/Muon/interface/HLTMuonTime.h"

// Collaborating Class Header
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/ModuleTiming.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TFile.h"
#include "TDirectory.h"
#include "TH1F.h"

using namespace std;
using namespace edm;


/// Constructor
HLTMuonTime::HLTMuonTime(const ParameterSet& pset)
{
  TimerIn=true;
  theTimerLabel=pset.getParameter<InputTag>("TimerLabel"); 
  ParameterSet ModulesForTiming =  pset.getUntrackedParameter<ParameterSet>("TimingModules");
  theMuonDigiModules=ModulesForTiming.getUntrackedParameter<vector<string> >("MuonDigiModules");
  theMuonLocalRecModules=ModulesForTiming.getUntrackedParameter<vector<string> >("MuonLocalRecModules");
  theMuonL2RecModules=ModulesForTiming.getUntrackedParameter<vector<string> >("MuonL2RecModules");
  theMuonL3RecModules=ModulesForTiming.getUntrackedParameter<vector<string> >("MuonL3RecModules");
  theMuonL2IsoModules=ModulesForTiming.getUntrackedParameter<vector<string> >("MuonL2IsoModules");
  theMuonL3IsoModules=ModulesForTiming.getUntrackedParameter<vector<string> >("MuonL3IsoModules");
  theTrackerDigiModules=ModulesForTiming.getUntrackedParameter<vector<string> >("TrackerDigiModules");
  theTrackerRecModules=ModulesForTiming.getUntrackedParameter<vector<string> >("TrackerRecModules");
  theCaloDigiModules=ModulesForTiming.getUntrackedParameter<vector<string> >("CaloDigiModules");
  theCaloRecModules=ModulesForTiming.getUntrackedParameter<vector<string> >("CaloRecModules");
  theTMax = pset.getParameter<double>("MaxTime");
  theNbins = pset.getParameter<unsigned int>("TimeNbins");
}

/// Destructor
HLTMuonTime::~HLTMuonTime(){
}

void HLTMuonTime::BookHistograms(){
  HistoDir=gDirectory;
  muondigi=HistoDir->mkdir("UnpackingMuon");
  muondigi->cd();
  for ( vector<string>::iterator its=theMuonDigiModules.begin();its!=theMuonDigiModules.end();++its)
    CreateHistograms("MuonDigis",*its);
  NumberOfModules.push_back(theMuonDigiModules.size());
  TDirs.push_back(muondigi);
  HistoDir->cd();
  CreateGlobalHistograms("MuonDigiTime","Muon Unpacking Time");

  trackerdigi=HistoDir->mkdir("UnpackingTracker");
  trackerdigi->cd();
  for ( vector<string>::iterator its=theTrackerDigiModules.begin();its!=theTrackerDigiModules.end();++its)
    CreateHistograms("TrackerDigis",*its);
  NumberOfModules.push_back(theTrackerDigiModules.size());
  TDirs.push_back(trackerdigi);
  HistoDir->cd();
  CreateGlobalHistograms("TrackerDigiTime", "Tracker Unpacking Time");

  trackerrec=HistoDir->mkdir("RecoTracker");
  trackerrec->cd();
  for ( vector<string>::iterator its=theTrackerRecModules.begin();its!=theTrackerRecModules.end();++its)
    CreateHistograms("TrackerRec",*its);
  NumberOfModules.push_back(theTrackerRecModules.size());
  TDirs.push_back(trackerrec);
  HistoDir->cd();
  CreateGlobalHistograms("TrackerRecTime", "Tracker Reconstruction Time");

  calodigi=HistoDir->mkdir("UnpackingCalo");
  calodigi->cd();
  for ( vector<string>::iterator its=theCaloDigiModules.begin();its!=theCaloDigiModules.end();++its)
    CreateHistograms("CaloDigis",*its);
  NumberOfModules.push_back(theCaloDigiModules.size());
  TDirs.push_back(calodigi);
  HistoDir->cd();
  CreateGlobalHistograms("CaloDigiTime", "Calo Unpacking Time");

  calorec=HistoDir->mkdir("RecCalo");
  calorec->cd();
  for ( vector<string>::iterator its=theCaloRecModules.begin();its!=theCaloRecModules.end();++its)
    CreateHistograms("CaloRec",*its);
  TDirs.push_back(calorec);
  NumberOfModules.push_back(theCaloRecModules.size());
  HistoDir->cd();
  CreateGlobalHistograms("CaloRecTime", "Calo Local Reconstruction Time");

  muonlocrec=HistoDir->mkdir("RecLocalMuon");
  muonlocrec->cd();
  for ( vector<string>::iterator its=theMuonLocalRecModules.begin();its!=theMuonLocalRecModules.end();++its)
    CreateHistograms("MuonLocalRec",*its);
  TDirs.push_back(muonlocrec);
  NumberOfModules.push_back(theMuonLocalRecModules.size());
  HistoDir->cd();
  CreateGlobalHistograms("MuonLocalRecTime", "Muon Local Reconstruction Time");

  muonl2rec=HistoDir->mkdir("RecL2Muon");
  muonl2rec->cd();
  for ( vector<string>::iterator its=theMuonL2RecModules.begin();its!=theMuonL2RecModules.end();++its)
    CreateHistograms("MuonL2Rec",*its);
  NumberOfModules.push_back(theMuonL2RecModules.size());
  TDirs.push_back(muonl2rec);
  HistoDir->cd();
  CreateGlobalHistograms("MuonL2RecTime", "Muon L2 Reconstruction Time");

  muonl3rec=HistoDir->mkdir("RecL3Muon");
  muonl3rec->cd();
  for ( vector<string>::iterator its=theMuonL3RecModules.begin();its!=theMuonL3RecModules.end();++its)
    CreateHistograms("MuonL3Rec",*its);
  NumberOfModules.push_back(theMuonL3RecModules.size());
  TDirs.push_back(muonl3rec);
  HistoDir->cd();
  CreateGlobalHistograms("MuonL3RecTime", "Muon L3 Reconstruction Time");

  muonl2iso=HistoDir->mkdir("RecL2Iso");
  muonl2iso->cd();
  for ( vector<string>::iterator its=theMuonL2IsoModules.begin();its!=theMuonL2IsoModules.end();++its)
    CreateHistograms("MuonL2Iso",*its);
  NumberOfModules.push_back(theMuonL2IsoModules.size());
  TDirs.push_back(muonl2iso);
  HistoDir->cd();
  CreateGlobalHistograms("MuonL2IsoTime", "Muon L2 Isolation Time");

  muonl3iso=HistoDir->mkdir("RecL3Iso");
  muonl3iso->cd();
  for ( vector<string>::iterator its=theMuonL3IsoModules.begin();its!=theMuonL3IsoModules.end();++its)
    CreateHistograms("MuonL3Iso",*its);
  TDirs.push_back(muonl3iso);
  NumberOfModules.push_back(theMuonL3IsoModules.size());
  HistoDir->cd();
  CreateGlobalHistograms("MuonL3IsoTime", "Muon L3 Isolation Time");
  CreateGlobalHistograms("HLTTime", "Global Time Requested Modules");
  return;
}


void HLTMuonTime::WriteHistograms(){
  if (!TimerIn) return;
  TDirs[0]->cd();
  unsigned int globalbin=0;
  unsigned int next=NumberOfModules[0];
  // Write the histos to file
  LogVerbatim ("HLTMuonVal")<<"Module Average Time when Run";
  for (unsigned int i=0; i<hTimes.size(); i++) {
    while ( i == next ) {
      ++globalbin;
      next+=NumberOfModules[globalbin];
      if (globalbin < TDirs.size())TDirs[globalbin]->cd();
    }
    hExclusiveTimes[i]->GetXaxis()->SetTitle("ms");
    hExclusiveTimes[i]->SetFillColor(38);
    hExclusiveTimes[i]->Write();
    LogVerbatim ("HLTMuonVal")<<ModuleNames[i]<<" "<<hExclusiveTimes[i]->GetMean()<<" rms:"<<hExclusiveTimes[i]->GetRMS()<<" ms";
    hTimes[i]->SetFillColor(46);
    hTimes[i]->GetXaxis()->SetTitle("ms");
    hTimes[i]->Write();
  }
  HistoDir->cd();
  LogVerbatim ("HLTMuonVal");
  LogVerbatim ("HLTMuonVal")<<"Global step times when Run";
  for (unsigned int i=0; i<hGlobalTimes.size(); i++) {
    hGlobalTimes[i]->SetFillColor(30);
    hGlobalTimes[i]->GetXaxis()->SetTitle("ms");
    hGlobalTimes[i]->Write();
    hExclusiveGlobalTimes[i]->SetFillColor(8);
    hExclusiveGlobalTimes[i]->GetXaxis()->SetTitle("ms");
    hExclusiveGlobalTimes[i]->Write();
    LogVerbatim("HLTMuonVal")<<hExclusiveGlobalTimes[i]->GetTitle()<<" "<<hExclusiveGlobalTimes[i]->GetMean()<<" rms:"<<hExclusiveGlobalTimes[i]->GetRMS()<<" ms";
  }
}

void HLTMuonTime::analyze(const Event & event ){
  if(!TimerIn)return;
  //reset the module times
  for (vector<double>::iterator reset=ModuleTime.begin();reset!=ModuleTime.end();++reset) *reset=0.;

  //get the timing info
  Handle<EventTime> evtTime;
  LogDebug("HLTMuonVal") << "About to try"<< theTimerLabel;  
  event.getByLabel(theTimerLabel, evtTime); 
  if (evtTime.failedToGet()){
    LogWarning("HLTMuonVal") << "!!!!!!!!! No timer run with label"<< theTimerLabel;
    TimerIn=false;
    return;
  }
  double total_time=0;
  unsigned size = evtTime->size();
  for(unsigned int i = 0; i != size; ++i){
      std::string module_name = evtTime->name(i);
      LogDebug("HLTMuonVal") << "Module name="<<module_name;
      for ( unsigned int j = 0; j != ModuleNames.size(); ++j ) {
	LogDebug("HLTMuonVal") << "ModuleNames["<<j<<"]="<<ModuleNames[j];
	if ( ModuleNames[j] == module_name) {
	  total_time+=evtTime->time(i);
	  ModuleTime[j]+=evtTime->time(i);
	  break;
	}
      }
  }
  
  unsigned int globalbin=0;
  unsigned int nbins=hGlobalTimes.size()-1;
  std::vector<float> GlobalTime;
  GlobalTime.reserve(nbins);
  unsigned int next=NumberOfModules[0];

  LogDebug("HLTMuonVal") << "next="<<next<<" and nextmax="<<hTimes.size()<<" Nbins="<<nbins;
  for (unsigned int i=0; i<hTimes.size(); i++) {
    LogDebug("HLTMuonVal") << "i="<<i;
    double t=1000*ModuleTime[i];
    hTimes[i]->Fill(t);
    if ( ModuleTime[i] > 0 ) hExclusiveTimes[i]->Fill(t);
    if ( i < next ) GlobalTime[globalbin]+=t;
    else while ( i == next ) {
      ++globalbin;
      next+=NumberOfModules[globalbin];
      LogDebug("HLTMuonVal") << "next="<<next;
      if ( i != next) GlobalTime[globalbin]=t;
      else  GlobalTime[globalbin]=0;
    }
  }  
  for ( unsigned int i=0; i<nbins; i++) {
    hGlobalTimes[i]->Fill(GlobalTime[i]);
    if (GlobalTime[i]>0)hExclusiveGlobalTimes[i]->Fill(GlobalTime[i]);
  }
  total_time*=1000;
  hGlobalTimes[nbins]->Fill(total_time);
  if (total_time>0)hExclusiveGlobalTimes[nbins]->Fill(total_time);
  return;
}

void HLTMuonTime::CreateHistograms(string Type, string Module)
{
  
  char chname[256];
  char chexname[256];
  char chtitle[256];
  char chextitle[256];
  TH1F* h;
  snprintf(chname, 255, "%s_%s",Type.c_str(),Module.c_str());
  snprintf(chexname, 255, "Exclusive_%s_%s",Type.c_str(),Module.c_str());
  snprintf(chtitle, 255, "Timing for %s (%s)",Module.c_str(),Type.c_str() );
  snprintf(chextitle, 255, "Exclusive Timing for %s (%s)",Module.c_str(),Type.c_str()); 
  h=new TH1F(chname, chtitle, theNbins, 0., theTMax);
  h->Sumw2();
  hTimes.push_back(h);    
  h=new TH1F(chexname, chextitle, theNbins, 0., theTMax);
  h->Sumw2();
  hExclusiveTimes.push_back(h);
  ModuleNames.push_back(Module);    
  ModuleTime.push_back(0.);
  return;
}
void HLTMuonTime::CreateGlobalHistograms(string Name, string Title)
{
  
  char chname[256];
  char chtitle[256];
  TH1F* h;
  h=new TH1F(Name.c_str(), Title.c_str(), theNbins, 0., theTMax);
  h->Sumw2();
  hGlobalTimes.push_back(h);
  snprintf(chname, 255, "Exclusive%s",Name.c_str());
  snprintf(chtitle, 255, "Exclusive %s",Title.c_str());
  h=new TH1F(chname, chtitle, theNbins, 0., theTMax);
  h->Sumw2();
  hExclusiveGlobalTimes.push_back(h);
  return;
}
