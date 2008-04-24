
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/04/16 19:49:42 $
 *  $Revision: 1.9 $
 *  \author G. Mila - INFN Torino
 */

#include "DQMOffline/Muon/src/MuonAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h" 
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
using namespace std;
using namespace edm;



MuonAnalyzer::MuonAnalyzer(const edm::ParameterSet& pSet) {

  cout<<"[MuonAnalyzer] Constructor called!"<<endl;

  parameters = pSet;

  // the services
  theService = new MuonServiceProxy(parameters.getParameter<ParameterSet>("ServiceParameters"));
  
  // STA Cosmic Muon Collection Label
  theSTACollectionLabel = parameters.getParameter<edm::InputTag>("CosmicsCollectionLabel");
  // Seeds Collection Label
  theSeedsCollectionLabel = parameters.getParameter<edm::InputTag>("seedsCollectionLabel");
  
  theMuEnergyAnalyzerFlag = parameters.getUntrackedParameter<bool>("DoMuonEnergyAnalysis",true);
  theSeedsAnalyzerFlag = parameters.getUntrackedParameter<bool>("DoMuonSeedAnalysis",true);
  theMuonRecoAnalyzerFlag = parameters.getUntrackedParameter<bool>("DoMuonRecoAnalysis",true);

  // do the analysis on muon energy
  if(theMuEnergyAnalyzerFlag)
    theMuEnergyAnalyzer = new MuonEnergyDepositAnalyzer(parameters.getParameter<ParameterSet>("muonEnergyAnalysis"), theService);
  // do the analysis on seeds
  if(theSeedsAnalyzerFlag)
    theSeedsAnalyzer = new MuonSeedsAnalyzer(parameters.getParameter<ParameterSet>("seedsAnalysis"), theService);
  // do the analysis on muon energy
  if(theMuonRecoAnalyzerFlag)
    theMuonRecoAnalyzer = new MuonRecoAnalyzer(parameters.getParameter<ParameterSet>("muonRecoAnalysis"), theService);

}

MuonAnalyzer::~MuonAnalyzer() { }


void MuonAnalyzer::beginJob(edm::EventSetup const& iSetup) {

  metname = "muonAnalyzer";

  LogTrace(metname)<<"[MuonAnalyzer] Parameters initialization";
  dbe = edm::Service<DQMStore>().operator->();
  dbe->setVerbose(1);

  if(theMuEnergyAnalyzerFlag) theMuEnergyAnalyzer->beginJob(iSetup, dbe);
  if(theSeedsAnalyzerFlag) theSeedsAnalyzer->beginJob(iSetup, dbe);
  if(theMuonRecoAnalyzerFlag) theMuonRecoAnalyzer->beginJob(iSetup, dbe);

}


void MuonAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  LogTrace(metname)<<"[MuonAnalyzer] Analysis of event # ";
  
  theService->update(iSetup);

   // Take the STA muon container
   edm::Handle<reco::MuonCollection> muons;
   iEvent.getByLabel(theSTACollectionLabel,muons);

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
  dbe->showDirStructure();
  bool outputMEsInRootFile = parameters.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = parameters.getParameter<std::string>("OutputFileName");
  if(outputMEsInRootFile){
    dbe->save(outputFileName);
  }
}

