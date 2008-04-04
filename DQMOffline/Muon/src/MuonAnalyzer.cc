
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/03/26 16:07:55 $
 *  $Revision: 1.4 $
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
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  dbe->setVerbose(1);
  
  // STA Cosmic Muon Collection Label
  theSTACollectionLabel = parameters.getParameter<edm::InputTag>("CosmicsCollectionLabel");
  // Seeds Collection Label
  theSeedsCollectionLabel = parameters.getParameter<edm::InputTag>("seedsCollectionLabel");
  
  // do the analysis on muon energy
  theMuEnergyAnalyzer = new MuonEnergyDepositAnalyzer(parameters.getParameter<ParameterSet>("muonEnergyAnalysis"), theService, dbe);
  // do the analysis on seeds
  theSeedsAnalyzer = new MuonSeedsAnalyzer(parameters.getParameter<ParameterSet>("seedsAnalysis"), theService, dbe);

}

MuonAnalyzer::~MuonAnalyzer() { }


void MuonAnalyzer::beginJob(edm::EventSetup const& iSetup) {

  metname = "muonAnalyzer";
  LogTrace(metname)<<"[MuonAnalyzer] Parameters initialization";

}


void MuonAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  LogTrace(metname)<<"[MuonAnalyzer] Analysis of event # ";

  theService->update(iSetup);

   // Take the STA muon container
   edm::Handle<reco::MuonCollection> muons;
   iEvent.getByLabel(theSTACollectionLabel,muons);

   for (reco::MuonCollection::const_iterator recoMu = muons->begin(); recoMu!=muons->end(); ++recoMu){
     LogTrace(metname)<<"[MuonAnalyzer] Call to the muon energy analyzer";
     theMuEnergyAnalyzer->analyze(iEvent, iSetup, *recoMu);
   }


   // Take the seeds container
   edm::Handle<TrajectorySeedCollection> seeds;
   iEvent.getByLabel(theSeedsCollectionLabel, seeds);

   for(TrajectorySeedCollection::const_iterator seed = seeds->begin(); seed != seeds->end(); ++seed){
     LogTrace(metname)<<"[MuonAnalyzer] Call to the seeds analyzer";
     theSeedsAnalyzer->analyze(iEvent, iSetup, *seed);
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

