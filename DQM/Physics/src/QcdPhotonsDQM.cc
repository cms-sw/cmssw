/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/06/26 15:24:28 $
 *  $Revision: 1.1 $
 *  \author Michael B. Anderson, University of Wisconsin Madison
 */

#include "DQM/Physics/src/QcdPhotonsDQM.h"

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

// Trigger stuff
#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include <vector>

#include <string>
#include <cmath>
using namespace std;
using namespace edm;
using namespace reco;



QcdPhotonsDQM::QcdPhotonsDQM(const ParameterSet& parameters) {
  // Get parameters from configuration file
  theTriggerPathToPass        = parameters.getParameter<string>("triggerPathToPass");
  theTriggerResultsCollection = parameters.getParameter<InputTag>("triggerResultsCollection");
  thePhotonCollectionLabel    = parameters.getParameter<InputTag>("photonCollection");
  theCaloJetCollectionLabel   = parameters.getParameter<InputTag>("caloJetCollection");
}

QcdPhotonsDQM::~QcdPhotonsDQM() { 
}


void QcdPhotonsDQM::beginJob(EventSetup const& iSetup) {
 
  logTraceName = "QcdPhotonAnalyzer";

  LogTrace(logTraceName)<<"Parameters initialization";
  theDbe = Service<DQMStore>().operator->();
 
  theDbe->setCurrentFolder("Physics/QcdPhotonsDQM");  // Use folder with name of PAG
  h_photon_et           = theDbe->book1D("h_photon_et",           "#gamma with highest E_{T};E_{T}(#gamma) (GeV)", 20, 0., 200.0);
  h_jet_et              = theDbe->book1D("h_jet_et",              "Jet with highest E_{T} (from "+theCaloJetCollectionLabel.label()+");E_{T}(jet) (GeV)",    20, 0., 200.0);
  h_deltaPhi_photon_jet = theDbe->book1D("h_deltaPhi_photon_jet", "#Delta#phi between Highest E_{T} #gamma and jet;#Delta#phi(#gamma,jet)", 20, 0, 3.1415926);
  h_deltaEt_photon_jet  = theDbe->book1D("h_deltaEt_photon_jet",  "(E_{T}(#gamma)-E_{T}(jet))/E_{T}(#gamma) when #Delta#phi(#gamma,jet) > 2.8;#DeltaE_{T}(#gamma,jet)/E_{T}(#gamma)", 20, -1.0, 1.0);
  h_jet_count           = theDbe->book1D("h_jet_count",           "Number of "+theCaloJetCollectionLabel.label()+" (E_{T} > 15 GeV);Number of Jets", 5, -0.5, 4.5);
  h_jet2_etOverPhotonEt = theDbe->book1D("h_jet2_etOverPhotonEt", "E_{T}(2nd highest jet) / E_{T}(#gamma);E_{T}(2nd Jet)/E_{T}(#gamma)", 20, 0.0, 4.0);
}


void QcdPhotonsDQM::analyze(const Event& iEvent, const EventSetup& iSetup) {

  LogTrace(logTraceName)<<"Analysis of event # ";

  // Did it pass certain HLT path?
  Handle<TriggerResults> HLTresults;
  iEvent.getByLabel(theTriggerResultsCollection, HLTresults); 
  HLTConfigProvider hltConfig;
  hltConfig.init("HLT");
  unsigned int triggerIndex = hltConfig.triggerIndex(theTriggerPathToPass);
  bool passed_HLT = HLTresults->accept(triggerIndex);
  if (!passed_HLT) return;


  // grab photons
  Handle<PhotonCollection> photonCollection;
  iEvent.getByLabel(thePhotonCollectionLabel, photonCollection);

  // If photon collection is empty, exit
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

  
  // If no decent photons, exit
  if (photon_et < 0.0) return;

  // Find the highest et jet
  Handle<CaloJetCollection> caloJetCollection;
  iEvent.getByLabel (theCaloJetCollectionLabel,caloJetCollection);

  float jet_et    = -9.0;
  float jet_eta   = -9.0;
  float jet_phi   = -9.0;
  int   jet_count = 0;
  float jet2_et   = -9.0;
  for (CaloJetCollection::const_iterator i_calojet = caloJetCollection->begin(); i_calojet != caloJetCollection->end(); i_calojet++) {

    float jet_current_et = i_calojet->et();

    // if it overlaps with photon, it is not a jet
    if ( fabs(i_calojet->eta()-photon_eta) < 0.2 && calcDeltaPhi(i_calojet->phi(), photon_phi) < 0.2) continue;
    // if it has too low Et, throw away
    if (jet_current_et < 15) continue;

    jet_count++;
    if (jet_current_et > jet_et) {
      jet_et  = i_calojet->et();
      jet_eta = i_calojet->eta();
      jet_phi = i_calojet->phi();
    } else if (jet_current_et > jet2_et) {
      jet2_et = i_calojet->et();
    }
  }

  // Fill histograms if photon & jet found
  if ( photon_et > 0.0 && jet_et > 0.0) {
    h_photon_et->Fill(photon_et);
    h_jet_et   ->Fill(jet_et);
    h_jet_count->Fill(jet_count);
    float deltaPhi = calcDeltaPhi(jet_phi, photon_phi);
    h_deltaPhi_photon_jet->Fill(deltaPhi);
    if (deltaPhi > 2.8) h_deltaEt_photon_jet->Fill( (photon_et-jet_et)/photon_et );
    if (jet2_et  > 0.0) h_jet2_etOverPhotonEt->Fill( jet2_et/photon_et );
  }
}


void QcdPhotonsDQM::endJob(void) {}


// This always returns only a positive deltaPhi
double QcdPhotonsDQM::calcDeltaPhi(double phi1, double phi2) {

  double deltaPhi = phi1 - phi2;

  if (deltaPhi < 0) deltaPhi = -deltaPhi;

  if (deltaPhi > 3.1415926) {
    deltaPhi = 2 * 3.1415926 - deltaPhi;
  }

  return deltaPhi;
}
