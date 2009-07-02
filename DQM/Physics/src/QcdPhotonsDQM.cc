/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/07/02 18:33:42 $
 *  $Revision: 1.7 $
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
  theMinCaloJetEt             = parameters.getParameter<int>("minCaloJetEt");
  theMinPhotonEt              = parameters.getParameter<int>("minPhotonEt");
}

QcdPhotonsDQM::~QcdPhotonsDQM() { 
}


void QcdPhotonsDQM::beginJob(EventSetup const& iSetup) {
 
  logTraceName = "QcdPhotonAnalyzer";

  LogTrace(logTraceName)<<"Parameters initialization";
  theDbe = Service<DQMStore>().operator->();
 
  theDbe->setCurrentFolder("Physics/QcdPhotons");  // Use folder with name of PAG

  std::stringstream aStringStream;
  std::string aString;
  aStringStream << theMinCaloJetEt;
  aString = aStringStream.str();

  // Keep the number of plots and number of bins to a minimum!
  h_photon_et           = theDbe->book1D("h_photon_et",     "#gamma with highest E_{T};E_{T}(#gamma) (GeV)", 20, 0., 200.0);
  h_photon_eta          = theDbe->book1D("h_photon_eta",    "#gamma with highest E_{T};#eta(#gamma)", 40, -5.0, 5.0);
  h_photon_phiMod       = theDbe->book1D("h_photon_phiMod", "#gamma with highest E_{T};#phi_{mod}=#phi#bullet180/#pi mod 20 - 10", 42, (-1.-1./20)*0.1745329, (1.+1./20.)*0.1745329 );
  h_jet_et              = theDbe->book1D("h_jet_et",        "Jet with highest E_{T} (from "+theCaloJetCollectionLabel.label()+");E_{T}(jet) (GeV)",    20, 0., 200.0);
  h_jet_eta             = theDbe->book1D("h_jet_eta",       "Jet with highest E_{T} (from "+theCaloJetCollectionLabel.label()+");#eta(jet)", 20, -5.0, 5.0);
  h_deltaPhi_photon_jet = theDbe->book1D("h_deltaPhi_photon_jet", "#Delta#phi between Highest E_{T} #gamma and jet;#Delta#phi(#gamma,jet)", 20, 0, 3.1415926);
  h_deltaEt_photon_jet  = theDbe->book1D("h_deltaEt_photon_jet",  "(E_{T}(#gamma)-E_{T}(jet))/E_{T}(#gamma) when #Delta#phi(#gamma,jet) > 2.8;#DeltaE_{T}(#gamma,jet)/E_{T}(#gamma)", 20, -1.0, 1.0);
  h_jet_count           = theDbe->book1D("h_jet_count",           "Number of "+theCaloJetCollectionLabel.label()+" (E_{T} > "+aString+" GeV);Number of Jets", 8, -0.5, 7.5);
  h_jet2_et             = theDbe->book1D("h_jet2_et",        "Jet with 2^{nd} highest E_{T} (from "+theCaloJetCollectionLabel.label()+");E_{T}(2^{nd} jet) (GeV)",    20, 0., 200.0);
  h_jet2_eta            = theDbe->book1D("h_jet2_eta", "Jet with 2^{nd} highest E_{T} (from "+theCaloJetCollectionLabel.label()+");#eta(2^{nd} jet)", 20, -5.0, 5.0);
  h_jet2_etOverPhotonEt = theDbe->book1D("h_jet2_etOverPhotonEt", "E_{T}(2^{nd} highest jet) / E_{T}(#gamma);E_{T}(2^{nd} Jet)/E_{T}(#gamma)", 20, 0.0, 4.0);
  h_deltaPhi_photon_jet2 = theDbe->book1D("h_deltaPhi_photon_jet2","#Delta#phi between Highest E_{T} #gamma and 2^{nd} highest jet;#Delta#phi(#gamma,2^{nd} jet)", 20, 0, 3.1415926);
  h_deltaR_jet_jet2      = theDbe->book1D("h_deltaR_jet_jet2", "#DeltaR between Highest Jet and 2^{nd} Highest;#DeltaR(1^{st} jet,2^{nd} jet)", 20, 0, 2*3.1415926);
}


void QcdPhotonsDQM::analyze(const Event& iEvent, const EventSetup& iSetup) {

  LogTrace(logTraceName)<<"Analysis of event # ";

  // Did it pass certain HLT path?
  Handle<TriggerResults> HLTresults;
  iEvent.getByLabel(theTriggerResultsCollection, HLTresults); 
  HLTConfigProvider hltConfig;
  hltConfig.init("HLT");
  unsigned int triggerIndex = hltConfig.triggerIndex(theTriggerPathToPass); // index of trigger path
  bool passed_HLT = true;
  if (triggerIndex < HLTresults->size()) passed_HLT = HLTresults->accept(triggerIndex);
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

    // Can't *really* determine if it's a photon when it's beyond eta of 2.5
    if ( fabs(recoPhoton->eta()) > 2.5 || recoPhoton->et() < theMinPhotonEt) continue;

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

  float jet_et    = -8.0;
  float jet_eta   = -8.0;
  float jet_phi   = -8.0;
  int   jet_count = 0;
  float jet2_et   = -9.0;
  float jet2_eta  = -9.0;
  float jet2_phi  = -9.0;
  for (CaloJetCollection::const_iterator i_calojet = caloJetCollection->begin(); i_calojet != caloJetCollection->end(); i_calojet++) {

    float jet_current_et = i_calojet->et();

    // if it overlaps with photon, it is not a jet
    if ( fabs(i_calojet->eta()-photon_eta) < 0.2 && calcDeltaPhi(i_calojet->phi(), photon_phi) < 0.2) continue;
    // if it has too low Et, throw away
    if (jet_current_et < theMinCaloJetEt) continue;

    jet_count++;
    if (jet_current_et > jet_et) {
      jet2_et  = jet_et;  // 2nd highest jet get's et from current highest
      jet2_eta = jet_eta;
      jet2_phi = jet_phi;
      jet_et   = i_calojet->et(); // current highest jet gets et from the new highest
      jet_eta  = i_calojet->eta();
      jet_phi  = i_calojet->phi();
    } else if (jet_current_et > jet2_et) {
      jet2_et  = i_calojet->et();
      jet2_eta = i_calojet->eta();
      jet2_phi = i_calojet->phi();
    }
  }


  ////////////////////////////////////////////////////////////////////
  // Fill histograms if photon & jet found
  if ( photon_et > 0.0 && jet_et > 0.0) {
    h_photon_et    ->Fill( photon_et  );
    h_photon_eta   ->Fill( photon_eta );
    h_photon_phiMod->Fill( fmod(photon_phi+3.14159,20.0*3.141592/180.0)-10.0*3.141592/180.0 );
    h_jet_et       ->Fill( jet_et     );
    h_jet_eta      ->Fill( jet_eta    );
    h_jet_count    ->Fill( jet_count  );
    float deltaPhi = calcDeltaPhi( jet_phi, photon_phi );
    h_deltaPhi_photon_jet->Fill( deltaPhi );
    if ( deltaPhi > 2.8 ) h_deltaEt_photon_jet->Fill( (photon_et-jet_et)/photon_et );
    if ( jet2_et  > 0.0 ) {
      h_jet2_et             ->Fill( jet2_et  );
      h_jet2_eta            ->Fill( jet2_eta );
      h_jet2_etOverPhotonEt ->Fill( jet2_et/photon_et );
      h_deltaPhi_photon_jet2->Fill( calcDeltaPhi(jet2_phi, photon_phi) );
      h_deltaR_jet_jet2     ->Fill( calcDeltaR(jet_eta,jet_phi,jet2_eta,jet2_phi) );
    }
  } 
  // End of Filling histograms
  ////////////////////////////////////////////////////////////////////
}


void QcdPhotonsDQM::endJob(void) {}

// Method for Calculating the delta-r between two things
float QcdPhotonsDQM::calcDeltaR(float eta1, float phi1, float eta2, float phi2) {

  float deltaEta = eta1 - eta2;
  float deltaPhi = calcDeltaPhi(phi1, phi2);

  float deltaRsqr = deltaEta*deltaEta + deltaPhi*deltaPhi;

  return sqrt(deltaRsqr);
} // End of calcDeltaR


// This always returns only a positive deltaPhi
float QcdPhotonsDQM::calcDeltaPhi(float phi1, float phi2) {

  float deltaPhi = phi1 - phi2;

  if (deltaPhi < 0) deltaPhi = -deltaPhi;

  if (deltaPhi > 3.1415926) {
    deltaPhi = 2 * 3.1415926 - deltaPhi;
  }

  return deltaPhi;
}
