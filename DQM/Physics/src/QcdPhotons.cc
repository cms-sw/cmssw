/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/06/24 11:13:00 $
 *  $Revision: 1.1 $
 *  \author Michael B. Anderson, University of Wisconsin Madison
 */

#include "DQM/Physics/src/QcdPhotons.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Physics Objects
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

#include <vector>

#include <string>
#include <cmath>
using namespace std;
using namespace edm;
using namespace reco;



QcdPhotons::QcdPhotons(const ParameterSet& parameters) {

  thePhotonCollectionLabel  = parameters.getParameter<InputTag>("photonCollection");
  theCaloJetCollectionLabel = parameters.getParameter<InputTag>("caloJetCollection");
}

QcdPhotons::~QcdPhotons() { 
}


void QcdPhotons::beginJob(EventSetup const& iSetup) {
 
  logTraceName = "QcdPhotonAnalyzer";

  LogTrace(logTraceName)<<"Parameters initialization";
  theDbe = Service<DQMStore>().operator->();
 
  theDbe->setCurrentFolder("Physics/QcdPhotons");  // Use folder with name of PAG
  h_photon_et           = theDbe->book1D("h_photon_et",           ";E_{T}(#gamma) (GeV)", 20, 0., 200.0);
  h_jet_et              = theDbe->book1D("h_jet_et",              ";E_{T}(jet) (GeV)",    20, 0., 200.0);
  h_deltaPhi_photon_jet = theDbe->book1D("h_deltaPhi_photon_jet", ";#Delta#phi(#gamma,jet)", 20, 0, 3.1415926);
}


void QcdPhotons::analyze(const Event& iEvent, const EventSetup& iSetup) {

  LogTrace(logTraceName)<<"Analysis of event # ";
  
  // grab photons
  Handle<PhotonCollection> photonCollection;
  iEvent.getByLabel(thePhotonCollectionLabel, photonCollection);

  // If no photons, exit
  if (!photonCollection.isValid()) return;

  // Find the highest et "decent" photon
  float photon_et  = -9.0;
  float photon_eta = -9.0;
  float photon_phi = -9.0;
  for (PhotonCollection::const_iterator recoPhoton = photonCollection->begin(); recoPhoton!=photonCollection->end(); recoPhoton++){
    // Require potential photon to pass some basic cuts
    if ( recoPhoton->ecalRecHitSumEtConeDR03() > 5+0.015*recoPhoton->et() ||
         recoPhoton->hcalTowerSumEtConeDR03()  > 10                       ||
         recoPhoton->hadronicOverEm()          > 0.5) continue;

    // Good photon found, store it
    photon_et  = recoPhoton->et();
    photon_eta = recoPhoton->eta();
    photon_phi = recoPhoton->phi();
    break;
  }


  // Find the highest et jet
  Handle<CaloJetCollection> caloJetCollection;
  iEvent.getByLabel (theCaloJetCollectionLabel,caloJetCollection);

  float jet_et  = -9.0;
  float jet_eta = -9.0;
  float jet_phi = -9.0;
  for (CaloJetCollection::const_iterator i_calojet = caloJetCollection->begin(); i_calojet != caloJetCollection->end(); i_calojet++){
    if (abs(i_calojet->eta() - photon_eta) < 0.2 && calcDeltaPhi(i_calojet->phi(), photon_phi) < 0.2) continue;

    jet_et  = i_calojet->et();
    jet_eta = i_calojet->eta();
    jet_phi = i_calojet->phi();
    break;
  }

  // Fill histograms
  h_photon_et->Fill(photon_et);
  h_jet_et->Fill(jet_et);
  h_deltaPhi_photon_jet->Fill(calcDeltaPhi(jet_phi, photon_phi));
}


void QcdPhotons::endJob(void) {}


// This always returns only a positive deltaPhi
double QcdPhotons::calcDeltaPhi(double phi1, double phi2) {

  double deltaPhi = phi1 - phi2;

  if (deltaPhi < 0) deltaPhi = -deltaPhi;

  if (deltaPhi > 3.1415926) {
    deltaPhi = 2 * 3.1415926 - deltaPhi;
  }

  return deltaPhi;
}
