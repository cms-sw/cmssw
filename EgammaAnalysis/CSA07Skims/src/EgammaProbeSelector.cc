
/** \class EgammaProbeSelector
 *
 *
 *  This class is an EDFilter for heavy W+jet and Z+jet events
 *
 *
 */

#include "EgammaAnalysis/CSA07Skims/interface/EgammaProbeSelector.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <TMath.h>

using namespace edm;
using namespace std;
using namespace reco;


EgammaProbeSelector::EgammaProbeSelector(const edm::ParameterSet& iConfig) {

  jetToken        = consumes<CaloJetCollection>(iConfig.getParameter<InputTag>("JetCollection"));
  minNumberOfjets = iConfig.getParameter<int>("MinNumberOfJets");
  jetEtMin        = iConfig.getParameter<double>("JetEtMin");
  jetEtaMin       = iConfig.getParameter<double>("JetEtaMin");
  jetEtaMax       = iConfig.getParameter<double>("JetEtaMax");

  nEvents         = 0;
  nSelectedEvents = 0;

  scToken          = consumes<SuperClusterCollection>(iConfig.getParameter<InputTag>("SuperClusterBarrelCollection"));
  scEEToken        = consumes<SuperClusterCollection>(iConfig.getParameter<InputTag>("SuperClusterEndCapCollection"));
  minNumberOfSuperClusters = iConfig.getParameter<int>("MinNumberOfSuperClusters");
  scEtMin          = iConfig.getParameter<double>("ScEtMin");
  scEtaMin         = iConfig.getParameter<double>("ScEtaMin");
  scEtaMax         = iConfig.getParameter<double>("ScEtaMax");
}


EgammaProbeSelector::~EgammaProbeSelector(){
  edm::LogVerbatim("EgammaProbeSelector")
    << " Number_events_read " << nEvents
    << " Number_events_kept " << nSelectedEvents
    << " Efficiency         " << ((double)nSelectedEvents)/((double) nEvents + 0.01) << std::endl;

}


bool EgammaProbeSelector::filter(edm::Event& iEvent, const edm::EventSetup& iSetup ){
  int nSC   = 0;
  int nJets = 0;
  nEvents++;

  // Get reconstructed Jets, check their Et and eta, and count their number
  Handle<CaloJetCollection> jetHandle;
  iEvent.getByToken(jetToken,jetHandle);

  if(jetHandle.isValid()){
    const reco::CaloJetCollection & jets = *(jetHandle.product());
    CaloJetCollection::const_iterator ijet;
    for(ijet = jets.begin(); ijet!= jets.end(); ijet++){
      if(ijet->pt()  > jetEtMin  &&
	 ijet->eta() > jetEtaMin &&
	 ijet->eta() < jetEtaMax )
	nJets++;
    }
  }

  // Get Super Clusters, check their Et and eta, and count their number
  // EB
  Handle<reco::SuperClusterCollection> scHandle;
  iEvent.getByToken(scToken,scHandle);

  if(scHandle.isValid()){
    const reco::SuperClusterCollection & SCs = *(scHandle.product());

    for(reco::SuperClusterCollection::const_iterator scIt = SCs.begin();
	scIt!= SCs.end(); scIt++){

      double scEt   = scIt->energy() * TMath::Sin(scIt->position().theta());
      double _eta = scIt->position().eta();
      if(scEt  > scEtMin && _eta > scEtaMin && _eta < scEtaMax)	nSC++;
    }
  }

  // EE
  Handle<reco::SuperClusterCollection> scEEHandle;
  iEvent.getByToken(scEEToken,scEEHandle);

  if(scEEHandle.isValid()){
    const reco::SuperClusterCollection & SCs = *(scEEHandle.product());
    for(reco::SuperClusterCollection::const_iterator scIt = SCs.begin();
	scIt!= SCs.end(); scIt++){

      double scEt   = scIt->energy() * TMath::Sin(scIt->position().theta());
      double _eta = scIt->position().eta();
      if(scEt  > scEtMin && _eta > scEtaMin && _eta < scEtaMax) nSC++;
    }
  }

  // Make final desicion on passed events
  bool accepted = false;
  if(nJets >= minNumberOfjets || nSC >=minNumberOfSuperClusters) {
    accepted = true;
    nSelectedEvents++;
  }

  return accepted;
}
