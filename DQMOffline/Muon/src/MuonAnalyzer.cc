
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/10/16 12:00:05 $
 *  $Revision: 1.29 $
 *  \author G. Mila - INFN Torino
 */

#include "DQMOffline/Muon/src/MuonAnalyzer.h"

#include "DQMOffline/Muon/src/MuonEnergyDepositAnalyzer.h"
#include "DQMOffline/Muon/src/MuonSeedsAnalyzer.h"
#include "DQMOffline/Muon/src/MuonRecoAnalyzer.h"
#include "DQMOffline/Muon/src/SegmentTrackAnalyzer.h"
#include "DQMOffline/Muon/src/MuonKinVsEtaAnalyzer.h"
#include "DQMOffline/Muon/src/DiMuonHistograms.h"
//#include "DQMOffline/Muon/src/MuonRecoOneHLT.h"
#include "DQMOffline/Muon/src/EfficiencyAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h" 
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
using namespace std;
using namespace edm;

MuonAnalyzer::MuonAnalyzer(const edm::ParameterSet& pSet) {
  parameters = pSet;
  
  // the services
  theService = new MuonServiceProxy(parameters.getParameter<ParameterSet>("ServiceParameters"));
  
  // Muon Collection Label
  theMuonCollectionLabel       = parameters.getParameter<edm::InputTag>("MuonCollection");
  theGlbMuTrackCollectionLabel = parameters.getParameter<edm::InputTag>("GlobalMuTrackCollection");
  theStaMuTrackCollectionLabel = parameters.getParameter<edm::InputTag>("STAMuTrackCollection");
  theSeedsCollectionLabel      = parameters.getParameter<edm::InputTag>("SeedCollection");
  theTriggerResultsLabel       = parameters.getParameter<edm::InputTag>("TriggerResultsLabel");
 
  // Analyzer Flags - define wheter or not run a submodule
  theMuEnergyAnalyzerFlag       = parameters.getUntrackedParameter<bool>("DoMuonEnergyAnalysis"   ,true);
  theSeedsAnalyzerFlag          = parameters.getUntrackedParameter<bool>("DoMuonSeedAnalysis"     ,true);
  theMuonRecoAnalyzerFlag       = parameters.getUntrackedParameter<bool>("DoMuonRecoAnalysis"     ,true);
  theMuonSegmentsAnalyzerFlag   = parameters.getUntrackedParameter<bool>("DoTrackSegmentsAnalysis",true);
  theMuonKinVsEtaAnalyzerFlag   = parameters.getUntrackedParameter<bool>("DoMuonKinVsEtaAnalysis" ,true);
  theDiMuonHistogramsFlag       = parameters.getUntrackedParameter<bool>("DoDiMuonHistograms"     ,true);
  //  theMuonRecoOneHLTAnalyzerFlag = parameters.getUntrackedParameter<bool>("DoMuonRecoOneHLT"       ,true);
  theEfficiencyAnalyzerFlag     = parameters.getUntrackedParameter<bool>("DoEfficiencyAnalysis"   ,true);
  
  // If the previous are defined... create the analyzer class 
  if(theMuEnergyAnalyzerFlag) 
    theMuEnergyAnalyzer = new MuonEnergyDepositAnalyzer(parameters.getParameter<ParameterSet>("muonEnergyAnalysis"), theService);
  if(theSeedsAnalyzerFlag)
    theSeedsAnalyzer = new MuonSeedsAnalyzer(parameters.getParameter<ParameterSet>("seedsAnalysis"), theService);
  if(theMuonRecoAnalyzerFlag)
    theMuonRecoAnalyzer = new MuonRecoAnalyzer(parameters.getParameter<ParameterSet>("muonRecoAnalysis"), theService);
  if(theMuonRecoAnalyzerFlag)
    theMuonKinVsEtaAnalyzer = new MuonKinVsEtaAnalyzer(parameters.getParameter<ParameterSet>("muonKinVsEtaAnalysis"), theService);
  if(theDiMuonHistogramsFlag)
    theDiMuonHistograms = new DiMuonHistograms(parameters.getParameter<ParameterSet>("dimuonHistograms"), theService);
  if(theMuonSegmentsAnalyzerFlag){
    // analysis on glb muon tracks
    ParameterSet  trackGlbMuAnalysisParameters = parameters.getParameter<ParameterSet>("trackSegmentsAnalysis");
    trackGlbMuAnalysisParameters.addParameter<edm::InputTag>("MuTrackCollection",theGlbMuTrackCollectionLabel);
    theGlbMuonSegmentsAnalyzer = new SegmentTrackAnalyzer(trackGlbMuAnalysisParameters, theService);
    // analysis on sta muon tracks
    ParameterSet  trackStaMuAnalysisParameters = parameters.getParameter<ParameterSet>("trackSegmentsAnalysis");
    trackStaMuAnalysisParameters.addParameter<edm::InputTag>("MuTrackCollection",theStaMuTrackCollectionLabel);
    theStaMuonSegmentsAnalyzer = new SegmentTrackAnalyzer(trackStaMuAnalysisParameters, theService);
  }
  //  if (theMuonRecoOneHLTAnalyzerFlag)
  //    theMuonRecoOneHLTAnalyzer = new MuonRecoOneHLT(parameters.getParameter<ParameterSet>("muonRecoOneHLTAnalysis"),theService);
  if(theEfficiencyAnalyzerFlag)
    theEfficiencyAnalyzer = new EfficiencyAnalyzer(parameters.getParameter<ParameterSet>("efficiencyAnalysis"), theService);
}

MuonAnalyzer::~MuonAnalyzer() {
  
  delete theService;
  if(theMuEnergyAnalyzerFlag)     delete theMuEnergyAnalyzer;
  if(theSeedsAnalyzerFlag)        delete theSeedsAnalyzer;
  if(theMuonRecoAnalyzerFlag)     delete theMuonRecoAnalyzer;
  if(theMuonSegmentsAnalyzerFlag) {
    delete theGlbMuonSegmentsAnalyzer;
    delete theStaMuonSegmentsAnalyzer;
  }
  if(theMuonKinVsEtaAnalyzerFlag)   delete theMuonKinVsEtaAnalyzer;
  if(theDiMuonHistogramsFlag)       delete theDiMuonHistograms;
  //  if(theMuonRecoOneHLTAnalyzerFlag) delete theMuonRecoOneHLTAnalyzer;
  if(theEfficiencyAnalyzerFlag)     delete theEfficiencyAnalyzer;
}
void MuonAnalyzer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup){
  //  if (theMuonRecoOneHLTAnalyzerFlag) theMuonRecoOneHLTAnalyzer->beginRun(iRun,iSetup);
}
void MuonAnalyzer::beginJob(void) {
  metname = "muonAnalyzer";
  
  LogTrace(metname)<<"[MuonAnalyzer] Parameters initialization";
  theDbe = edm::Service<DQMStore>().operator->();
  
  if(theMuEnergyAnalyzerFlag)       theMuEnergyAnalyzer->beginJob(theDbe);
  if(theSeedsAnalyzerFlag)          theSeedsAnalyzer->beginJob(theDbe);
  if(theMuonRecoAnalyzerFlag)       theMuonRecoAnalyzer->beginJob(theDbe);
  if(theMuonSegmentsAnalyzerFlag)   theGlbMuonSegmentsAnalyzer->beginJob(theDbe);
  if(theMuonSegmentsAnalyzerFlag)   theStaMuonSegmentsAnalyzer->beginJob(theDbe);
  if(theMuonKinVsEtaAnalyzerFlag)   theMuonKinVsEtaAnalyzer->beginJob(theDbe);
  if(theDiMuonHistogramsFlag)       theDiMuonHistograms->beginJob(theDbe);
  //  if(theMuonRecoOneHLTAnalyzerFlag) theMuonRecoOneHLTAnalyzer->beginJob(theDbe);
  if(theEfficiencyAnalyzerFlag)     theEfficiencyAnalyzer->beginJob(theDbe); 
}
void MuonAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  LogTrace(metname)<<"[MuonAnalyzer] Analysis of event # ";
  theService->update(iSetup);
  
  // Take the STA muon container
  edm::Handle<reco::MuonCollection> muons;
  iEvent.getByLabel(theMuonCollectionLabel,muons);

  edm::Handle<TriggerResults> triggerResults;
  iEvent.getByLabel(theTriggerResultsLabel, triggerResults);
  
  if(muons.isValid()){
    for (reco::MuonCollection::const_iterator recoMu = muons->begin(); recoMu!=muons->end(); ++recoMu){
      if(theMuEnergyAnalyzerFlag){
	LogTrace(metname)<<"[MuonAnalyzer] Call to the muon energy analyzer";
	theMuEnergyAnalyzer->analyze(iEvent, iSetup, *recoMu);
      }
      if(theMuonRecoAnalyzerFlag){
	LogTrace(metname)<<"[MuonAnalyzer] Call to the muon reco analyzer";
	theMuonRecoAnalyzer->analyze(iEvent, iSetup, *recoMu);
      }
      if(theMuonKinVsEtaAnalyzerFlag){
	LogTrace(metname)<<"[MuonAnalyzer] Call to the muon KinVsEta analyzer";
	theMuonKinVsEtaAnalyzer->analyze(iEvent, iSetup, *recoMu);
      }
      //      if(theMuonRecoOneHLTAnalyzerFlag) {
      //	LogTrace(metname)<<"[MuonAnalyzer] Call to the muon reco One HLT analyzer";
	//	theMuonRecoOneHLTAnalyzer->analyze(iEvent, iSetup, *recoMu, *triggerResults);
      //	theMuonRecoOneHLTAnalyzer->analyze(iEvent, iSetup, *triggerResults);
      //      }
    }
    //    if(theMuonRecoOneHLTAnalyzerFlag) {
    //      LogTrace(metname)<<"[MuonAnalyzer] Call to the muon reco One HLT analyzer";
    //      theMuonRecoOneHLTAnalyzer->analyze(iEvent, iSetup, *triggerResults);
    //    }
    if (theEfficiencyAnalyzerFlag){
      LogTrace(metname)<<"[MuonAnalyzer] Call to the efficiency analyzer";
      theEfficiencyAnalyzer->analyze(iEvent,iSetup);
    }
    if (theDiMuonHistogramsFlag){
      LogTrace(metname)<<"[MuonAnalyzer] Call to the dimuon analyzer";
      theDiMuonHistograms->analyze(iEvent,iSetup);
    }
  }
  
   // Take the track containers
   Handle<reco::TrackCollection> glbTracks;
   iEvent.getByLabel(theGlbMuTrackCollectionLabel,glbTracks);
   Handle<reco::TrackCollection> staTracks;
   iEvent.getByLabel(theStaMuTrackCollectionLabel,staTracks);

   if(glbTracks.isValid()){
     for (reco::TrackCollection::const_iterator recoTrack = glbTracks->begin(); recoTrack!=glbTracks->end(); ++recoTrack){
       if(theMuonSegmentsAnalyzerFlag){
	 LogTrace(metname)<<"[SegmentsAnalyzer] Call to the track segments analyzer for glb muons";
	 theGlbMuonSegmentsAnalyzer->analyze(iEvent, iSetup, *recoTrack);
       }
     }
   }
   if(staTracks.isValid()){
     for (reco::TrackCollection::const_iterator recoTrack = staTracks->begin(); recoTrack!=staTracks->end(); ++recoTrack){
       if(theMuonSegmentsAnalyzerFlag){
	 LogTrace(metname)<<"[SegmentsAnalyzer] Call to the track segments analyzer for sta muons";
	 theStaMuonSegmentsAnalyzer->analyze(iEvent, iSetup, *recoTrack);
       }
     }
   }
   
   
     

   // Take the seeds container
   edm::Handle<TrajectorySeedCollection> seeds;
   iEvent.getByLabel(theSeedsCollectionLabel, seeds);
   if(seeds.isValid()){
     for(TrajectorySeedCollection::const_iterator seed = seeds->begin(); seed != seeds->end(); ++seed){
       if(theSeedsAnalyzerFlag){
	 LogTrace(metname)<<"[MuonAnalyzer] Call to the seeds analyzer";
	 theSeedsAnalyzer->analyze(iEvent, iSetup, *seed);
       }
     }
   }

}


void MuonAnalyzer::endJob(void) {
  LogTrace(metname)<<"[MuonAnalyzer] Saving the histos";
  bool outputMEsInRootFile = parameters.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = parameters.getParameter<std::string>("OutputFileName");
  if(outputMEsInRootFile){
    theDbe->showDirStructure();
    theDbe->save(outputFileName);
  }
}

