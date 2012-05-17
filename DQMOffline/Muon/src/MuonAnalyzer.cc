
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/01/22 18:42:41 $
 *  $Revision: 1.23 $
 *  \author G. Mila - INFN Torino
 */

#include "DQMOffline/Muon/src/MuonAnalyzer.h"

#include "DQMOffline/Muon/src/MuonEnergyDepositAnalyzer.h"
#include "DQMOffline/Muon/src/MuonSeedsAnalyzer.h"
#include "DQMOffline/Muon/src/MuonRecoAnalyzer.h"
#include "DQMOffline/Muon/src/SegmentTrackAnalyzer.h"
#include "DQMOffline/Muon/src/MuonKinVsEtaAnalyzer.h"
#include "DQMOffline/Muon/src/DiMuonHistograms.h"


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
  theMuonCollectionLabel = parameters.getParameter<edm::InputTag>("MuonCollection");
  // Glb Muon Collection Label
  theGlbMuTrackCollectionLabel = parameters.getParameter<edm::InputTag>("GlobalMuTrackCollection");
  // STA Muon Collection Label
  theStaMuTrackCollectionLabel = parameters.getParameter<edm::InputTag>("STAMuTrackCollection");
  // Seeds Collection Label
  theSeedsCollectionLabel = parameters.getParameter<edm::InputTag>("SeedCollection");
  
  theMuEnergyAnalyzerFlag = parameters.getUntrackedParameter<bool>("DoMuonEnergyAnalysis",true);
  theSeedsAnalyzerFlag = parameters.getUntrackedParameter<bool>("DoMuonSeedAnalysis",true);
  theMuonRecoAnalyzerFlag = parameters.getUntrackedParameter<bool>("DoMuonRecoAnalysis",true);
  theMuonSegmentsAnalyzerFlag = parameters.getUntrackedParameter<bool>("DoTrackSegmentsAnalysis",true);
  theMuonKinVsEtaAnalyzerFlag = parameters.getUntrackedParameter<bool>("DoMuonKinVsEtaAnalysis",false);
  theDiMuonHistogramsFlag = parameters.getUntrackedParameter<bool>("DoDiMuonHistograms",true);
  
  // do the analysis on muon energy
  if(theMuEnergyAnalyzerFlag)
    theMuEnergyAnalyzer = new MuonEnergyDepositAnalyzer(parameters.getParameter<ParameterSet>("muonEnergyAnalysis"), theService);
  // do the analysis on seeds
  if(theSeedsAnalyzerFlag)
    theSeedsAnalyzer = new MuonSeedsAnalyzer(parameters.getParameter<ParameterSet>("seedsAnalysis"), theService);
  // do the analysis on muon energy
  if(theMuonRecoAnalyzerFlag)
    theMuonRecoAnalyzer = new MuonRecoAnalyzer(parameters.getParameter<ParameterSet>("muonRecoAnalysis"), theService);
  // do the analysis of kin quantities in eta regions
  if(theMuonRecoAnalyzerFlag)
    theMuonKinVsEtaAnalyzer = new MuonKinVsEtaAnalyzer(parameters.getParameter<ParameterSet>("muonKinVsEtaAnalysis"), theService);
  if(theDiMuonHistogramsFlag)
    theDiMuonHistograms = new DiMuonHistograms(parameters.getParameter<ParameterSet>("dimuonHistograms"), theService);
  // do the analysis on muon track segments
  if(theMuonSegmentsAnalyzerFlag){
    // analysis on glb muon tracks
    ParameterSet  trackGlbMuAnalysisParameters = parameters.getParameter<ParameterSet>("trackSegmentsAnalysis");
    trackGlbMuAnalysisParameters.addParameter<edm::InputTag>("MuTrackCollection",
							     theGlbMuTrackCollectionLabel);
    theGlbMuonSegmentsAnalyzer = new SegmentTrackAnalyzer(trackGlbMuAnalysisParameters, theService);
    // analysis on sta muon tracks
    ParameterSet  trackStaMuAnalysisParameters = parameters.getParameter<ParameterSet>("trackSegmentsAnalysis");
    trackStaMuAnalysisParameters.addParameter<edm::InputTag>("MuTrackCollection",
							     theStaMuTrackCollectionLabel);
    theStaMuonSegmentsAnalyzer = new SegmentTrackAnalyzer(trackStaMuAnalysisParameters, theService);
  }
}

MuonAnalyzer::~MuonAnalyzer() {
 
  delete theService;
  if(theMuEnergyAnalyzerFlag) delete theMuEnergyAnalyzer;
  if(theSeedsAnalyzerFlag) delete theSeedsAnalyzer;
  if(theMuonRecoAnalyzerFlag) delete theMuonRecoAnalyzer;
  if(theMuonSegmentsAnalyzerFlag) {
    delete theGlbMuonSegmentsAnalyzer;
    delete theStaMuonSegmentsAnalyzer;
  }
  if(theMuonKinVsEtaAnalyzerFlag) {
    delete theMuonKinVsEtaAnalyzer;
  }
  if(theDiMuonHistogramsFlag) {
    delete theDiMuonHistograms;
  }
}


void MuonAnalyzer::beginJob(void) {

  metname = "muonAnalyzer";

  LogTrace(metname)<<"[MuonAnalyzer] Parameters initialization";
  theDbe = edm::Service<DQMStore>().operator->();
 
  if(theMuEnergyAnalyzerFlag) theMuEnergyAnalyzer->beginJob(theDbe);
  if(theSeedsAnalyzerFlag) theSeedsAnalyzer->beginJob(theDbe);
  if(theMuonRecoAnalyzerFlag) theMuonRecoAnalyzer->beginJob(theDbe);
  if(theMuonSegmentsAnalyzerFlag) {
    theGlbMuonSegmentsAnalyzer->beginJob(theDbe);
    theStaMuonSegmentsAnalyzer->beginJob(theDbe);
  }
  if(theMuonKinVsEtaAnalyzerFlag) {
    theMuonKinVsEtaAnalyzer->beginJob(theDbe);
  }

  if(theDiMuonHistogramsFlag) {
    theDiMuonHistograms->beginJob(theDbe);
  }

}


void MuonAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  LogTrace(metname)<<"[MuonAnalyzer] Analysis of event # ";
  
  theService->update(iSetup);

   // Take the STA muon container
   edm::Handle<reco::MuonCollection> muons;
   iEvent.getByLabel(theMuonCollectionLabel,muons);

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

