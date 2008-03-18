
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/03/13 11:11:12 $
 *  $Revision: 1.1 $
 *  \author G. Mila - INFN Torino
 */

#include "DQMOffline/Muon/src/MuonAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h" 

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
using namespace std;
using namespace edm;



MuonAnalyzer::MuonAnalyzer(const edm::ParameterSet& pSet) {

  cout<<"[MuonAnalyzer] Constructor called!"<<endl;

  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  dbe->setVerbose(1);
  parameters = pSet;

  // STA Cosmic Muon Collection Label
  theSTACollectionLabel = parameters.getParameter<edm::InputTag>("CosmicsCollectionLabel");
  
  // do the analysis on muon energy
  theMuEnergyAnalyzer = new MuonEnergyDepositAnalyzer(parameters.getParameter<ParameterSet>("muonEnergyAnalysis"));

}

MuonAnalyzer::~MuonAnalyzer() { }


void MuonAnalyzer::beginJob(edm::EventSetup const& iSetup) {

  cout<<"[MuonAnalyzer] Parameters initialization"<<endl;

  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  dbe->setVerbose(1);

  theMuEnergyAnalyzer->beginJob(iSetup, dbe);

}


void MuonAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  cout<<"[MuonAnalyzer] Analysis of event # "<<endl;

   // Take the STA muon container
   edm::Handle<reco::MuonCollection> muons;
   iEvent.getByLabel(theSTACollectionLabel,muons);

   for (reco::MuonCollection::const_iterator recoMu = muons->begin(); recoMu!=muons->end(); ++recoMu){
     
     cout<<"[MuonAnalyzer] Call to the muon energy analyzer"<<endl;
     theMuEnergyAnalyzer->analyze(iEvent, iSetup, *recoMu);

   }

}


void MuonAnalyzer::endJob(void) {
  cout<<"[MuonAnalyzer] Saving the histos"<<endl;
  dbe->showDirStructure();
  bool outputMEsInRootFile = parameters.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = parameters.getParameter<std::string>("OutputFileName");
  if(outputMEsInRootFile){
    dbe->save(outputFileName);
  }
}

