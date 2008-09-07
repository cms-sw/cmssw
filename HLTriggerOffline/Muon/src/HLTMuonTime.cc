/** \class HLTMuonTimeTime
 *  Get L1/HLT efficiency/rate plots
 *
 *  \author M. Vander Donckt  
 */
//
#include "HLTriggerOffline/Muon/interface/HLTMuonTime.h"

// Collaborating Class Header
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/ModuleTiming.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


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
  dbe = NULL;
  if (pset.getUntrackedParameter < bool > ("DQMStore", false)) {
    dbe = Service < DQMStore > ().operator->();
    dbe->setVerbose(0);
  }
  bool disable =pset.getUntrackedParameter < bool > ("disableROOToutput", false);
  if (disable) theRootFileName="";
  else theRootFileName = pset.getUntrackedParameter<std::string>("RootFileName");
  
}

/// Destructor
HLTMuonTime::~HLTMuonTime(){
}

void HLTMuonTime::BookHistograms(){
  LogDebug("HLTMuonVal")<<"directory "<<dbe->pwd();
  dbe->setCurrentFolder("HLT/Muon/Timing");
  LogDebug("HLTMuonVal")<<"directory "<<dbe->pwd();
  if (theMuonDigiModules.size()){
    dbe->setCurrentFolder("HLT/Muon/Timing/UnpackingMuon");
    LogDebug("HLTMuonVal")<<"directory "<<dbe->pwd();
    for ( vector<string>::iterator its=theMuonDigiModules.begin();its!=theMuonDigiModules.end();++its)
      CreateHistograms("MuonDigis",*its);
    NumberOfModules.push_back(theMuonDigiModules.size());
    TDirs.push_back("HLT/Muon/Timing/UnpackingMuon");
    dbe->setCurrentFolder("HLT/Muon/Timing");
    LogDebug("HLTMuonVal")<<"directory "<<dbe->pwd();
    CreateGlobalHistograms("MuonDigiTime","Muon Unpacking Time");
  }
  if (theTrackerDigiModules.size()){
    dbe->setCurrentFolder("HLT/Muon/Timing/UnpackingTracker");
    for ( vector<string>::iterator its=theTrackerDigiModules.begin();its!=theTrackerDigiModules.end();++its)
      CreateHistograms("TrackerDigis",*its);
    NumberOfModules.push_back(theTrackerDigiModules.size());
    TDirs.push_back("HLT/Muon/Timing/UnpackingTracker");
    dbe->setCurrentFolder("HLT/Muon/Timing");
    CreateGlobalHistograms("TrackerDigiTime", "Tracker Unpacking Time");
  }
  if (theTrackerRecModules.size()){
    dbe->setCurrentFolder("HLT/Muon/Timing/RecoTracker");
    for ( vector<string>::iterator its=theTrackerRecModules.begin();its!=theTrackerRecModules.end();++its)
      CreateHistograms("TrackerRec",*its);
    NumberOfModules.push_back(theTrackerRecModules.size());
    TDirs.push_back("HLT/Muon/Timing/RecoTracker");
    dbe->setCurrentFolder("HLT/Muon/Timing");
    CreateGlobalHistograms("TrackerRecTime", "Tracker Reconstruction Time");
  }
  if (theCaloDigiModules.size()){
    dbe->setCurrentFolder("HLT/Muon/Timing/UnpackingCalo");
    for ( vector<string>::iterator its=theCaloDigiModules.begin();its!=theCaloDigiModules.end();++its)
      CreateHistograms("CaloDigis",*its);
    NumberOfModules.push_back(theCaloDigiModules.size());
    TDirs.push_back("HLT/Muon/Timing/UnpackingCalo");
    dbe->setCurrentFolder("HLT/Muon/Timing");
    CreateGlobalHistograms("CaloDigiTime", "Calo Unpacking Time");
  }
  if ( theCaloRecModules.size() ) {
    dbe->setCurrentFolder("HLT/Muon/Timing/RecCalo");
    for ( vector<string>::iterator its=theCaloRecModules.begin();its!=theCaloRecModules.end();++its)
      CreateHistograms("CaloRec",*its);
    TDirs.push_back("HLT/Muon/Timing/RecCalo");
    NumberOfModules.push_back(theCaloRecModules.size());
    dbe->setCurrentFolder("HLT/Muon/Timing");
    CreateGlobalHistograms("CaloRecTime", "Calo Local Reconstruction Time");
  }
  if (theMuonLocalRecModules.size()){
    dbe->setCurrentFolder("HLT/Muon/Timing/RecLocalMuon");
    for ( vector<string>::iterator its=theMuonLocalRecModules.begin();its!=theMuonLocalRecModules.end();++its)
      CreateHistograms("MuonLocalRec",*its);
    TDirs.push_back("HLT/Muon/Timing/RecLocalMuon");
    NumberOfModules.push_back(theMuonLocalRecModules.size());
    dbe->setCurrentFolder("HLT/Muon/Timing");
    CreateGlobalHistograms("MuonLocalRecTime", "Muon Local Reconstruction Time");
  }
  if (  theMuonL2RecModules.size() ){
    dbe->setCurrentFolder("HLT/Muon/Timing/RecL2Muon");
    for ( vector<string>::iterator its=theMuonL2RecModules.begin();its!=theMuonL2RecModules.end();++its)
      CreateHistograms("MuonL2Rec",*its);
    NumberOfModules.push_back(theMuonL2RecModules.size());
    TDirs.push_back("HLT/Muon/Timing/RecL2Muon");
    dbe->setCurrentFolder("HLT/Muon/Timing");
    CreateGlobalHistograms("MuonL2RecTime", "Muon L2 Reconstruction Time");
  }
  if ( theMuonL3RecModules.size() ) {
    dbe->setCurrentFolder("HLT/Muon/Timing/RecL3Muon");
    for ( vector<string>::iterator its=theMuonL3RecModules.begin();its!=theMuonL3RecModules.end();++its)
      CreateHistograms("MuonL3Rec",*its);
    NumberOfModules.push_back(theMuonL3RecModules.size());
    TDirs.push_back("HLT/Muon/Timing/RecL3Muon");
    dbe->setCurrentFolder("HLT/Muon/Timing");
    CreateGlobalHistograms("MuonL3RecTime", "Muon L3 Reconstruction Time");
  }
  if (theMuonL2IsoModules.size()){
  dbe->setCurrentFolder("HLT/Muon/Timing/RecL2Iso");
  for ( vector<string>::iterator its=theMuonL2IsoModules.begin();its!=theMuonL2IsoModules.end();++its)
    CreateHistograms("MuonL2Iso",*its);
  NumberOfModules.push_back(theMuonL2IsoModules.size());
  TDirs.push_back("HLT/Muon/Timing/RecL2Iso");
  dbe->setCurrentFolder("HLT/Muon/Timing");
  CreateGlobalHistograms("MuonL2IsoTime", "Muon L2 Isolation Time");
  }
  if(theMuonL3IsoModules.size()){
    dbe->setCurrentFolder("HLT/Muon/Timing/RecL3Iso");
    for ( vector<string>::iterator its=theMuonL3IsoModules.begin();its!=theMuonL3IsoModules.end();++its)
      CreateHistograms("MuonL3Iso",*its);
    TDirs.push_back("HLT/Muon/Timing/RecL3Iso");
    NumberOfModules.push_back(theMuonL3IsoModules.size());
    dbe->setCurrentFolder("HLT/Muon/Timing");
    CreateGlobalHistograms("MuonL3IsoTime", "Muon L3 Isolation Time");
  }
  dbe->setCurrentFolder("HLT/Muon/Timing");
  CreateGlobalHistograms("HLTTime", "Global Time Requested Modules");
  return;
}


void HLTMuonTime::WriteHistograms(){
  if (!TimerIn) return;
  unsigned int globalbin=0;
  unsigned int next=NumberOfModules[0];
  // Write the histos to file
  LogVerbatim ("HLTMuonVal")<<"Module Average Time when Run";
  for (unsigned int i=0; i<hTimes.size(); i++) {
    while ( i == next ) {
      ++globalbin;
      next+=NumberOfModules[globalbin];
    }
    hExclusiveTimes[i]->setAxisTitle("ms");
    LogVerbatim ("HLTMuonVal")<<ModuleNames[i]<<" "<<hExclusiveTimes[i]->getMean()<<" rms:"<<hExclusiveTimes[i]->getRMS()<<" ms";
    hTimes[i]->setAxisTitle("ms",1);
  }
  LogVerbatim ("HLTMuonVal");
  LogVerbatim ("HLTMuonVal")<<"Global step times when Run";
  for (unsigned int i=0; i<hGlobalTimes.size(); i++) {
    hGlobalTimes[i]->setAxisTitle("ms",1);
    hExclusiveGlobalTimes[i]->setAxisTitle("ms",1);
    LogVerbatim("HLTMuonVal")<<hExclusiveGlobalTimes[i]->getTitle()<<" "<<hExclusiveGlobalTimes[i]->getMean()<<" rms:"<<hExclusiveGlobalTimes[i]->getRMS()<<" ms";
  }
  if (theRootFileName.size() != 0 && dbe) dbe->save(theRootFileName);

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
    LogDebug("HLTMuonVal") << "!!!!!!!!! No timer run with label"<< theTimerLabel;
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
  LogDebug("HLTMuonVal")<<"directory "<<dbe->pwd()<<" Name:"<<chname;

  hTimes.push_back(dbe->book1D(chname, h));
  delete h;   
  h=new TH1F(chexname, chextitle, theNbins, 0., theTMax);
  h->Sumw2();
  hExclusiveTimes.push_back(dbe->book1D(chexname, h));
  delete h;
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
  hGlobalTimes.push_back(dbe->book1D(Name, h));
  delete h;
  snprintf(chname, 255, "Exclusive%s",Name.c_str());
  snprintf(chtitle, 255, "Exclusive %s",Title.c_str());
  h=new TH1F(chname, chtitle, theNbins, 0., theTMax);
  h->Sumw2();
  hExclusiveGlobalTimes.push_back(dbe->book1D(Name, h));
  delete h;
  return;
}
