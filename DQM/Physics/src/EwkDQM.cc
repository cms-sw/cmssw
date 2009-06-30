/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/06/28 09:46:43 $
 *  $Revision: 1.2 $
 *  \author Michael B. Anderson, University of Wisconsin Madison
 */

#include "DQM/Physics/src/EwkDQM.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Physics Objects
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

// Trigger stuff
#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "TLorentzVector.h"

#include <vector>

#include <string>
#include <cmath>
using namespace std;
using namespace edm;
using namespace reco;



EwkDQM::EwkDQM(const ParameterSet& parameters) {
  // Get parameters from configuration file
  theTriggerPathToPass        = parameters.getParameter<string>("triggerPathToPass");
  theTriggerResultsCollection = parameters.getParameter<InputTag>("triggerResultsCollection");
  theMuonCollectionLabel      = parameters.getParameter<InputTag>("muonCollection");
  theElectronCollectionLabel  = parameters.getParameter<InputTag>("electronCollection");
  theCaloJetCollectionLabel   = parameters.getParameter<InputTag>("caloJetCollection");
}

EwkDQM::~EwkDQM() { 
}


void EwkDQM::beginJob(EventSetup const& iSetup) {
 
  logTraceName = "EwkAnalyzer";

  LogTrace(logTraceName)<<"Parameters initialization";
  theDbe = Service<DQMStore>().operator->();
 
  theDbe->setCurrentFolder("Physics/EwkDQM");  // Use folder with name of PAG

  // Keep the number of plots and number of bins to a minimum!
  h_mumu_invMass = theDbe->book1D("h_mumu_invMass", "#mu#mu Invariant Mass;InvMass (GeV)", 20, 40.0, 140.0);
  h_ee_invMass   = theDbe->book1D("h_ee_invMass",   "ee Invariant Mass;InvMass (Gev)"    , 20, 40.0, 140.0);
  h_jet_et       = theDbe->book1D("h_jet_et",       "Jet with highest E_{T} (from "+theCaloJetCollectionLabel.label()+");E_{T}(jet) (GeV)",    20, 0., 200.0);
  h_jet_count    = theDbe->book1D("h_jet_count",    "Number of "+theCaloJetCollectionLabel.label()+" (E_{T} > 15 GeV);Number of Jets", 8, -0.5, 7.5);
  
}


void EwkDQM::analyze(const Event& iEvent, const EventSetup& iSetup) {

  LogTrace(logTraceName)<<"Analysis of event # ";

  // Did it pass certain HLT path?
  Handle<TriggerResults> HLTresults;
  iEvent.getByLabel(theTriggerResultsCollection, HLTresults); 
  HLTConfigProvider hltConfig;
  hltConfig.init("HLT");
  unsigned int triggerIndex = hltConfig.triggerIndex(theTriggerPathToPass);
  bool passed_HLT = true;
  if (triggerIndex < HLTresults->size()) passed_HLT = HLTresults->accept(triggerIndex);
  //if (!passed_HLT) return;


  ////////////////////////////////////////////////////////////////////////////////
  // grab "gaussian sum fitting" electrons
  Handle<GsfElectronCollection> electronCollection;
  iEvent.getByLabel(theElectronCollectionLabel, electronCollection);

  // Find the electron pair closest to z mass
  float zMass = 91.1876;
  float closestEEInvMasstoZFound = -9.0;
  float electron_et  = -9.0;
  float electron_eta = -9.0;
  float electron_phi = -9.0;
  float electron2_et  = -9.0;
  float electron2_eta = -9.0;
  float electron2_phi = -9.0;

  if(electronCollection.isValid()){
   for (reco::GsfElectronCollection::const_iterator recoElectron=electronCollection->begin(); recoElectron!=electronCollection->end(); recoElectron++){

     // Require electron to pass some basic cuts
     if (recoElectron->et() < 20 && fabs(recoElectron->eta())<2.5) continue;

    // loop over all the other electrons
     for (reco::GsfElectronCollection::const_iterator recoElectron2=recoElectron+1; recoElectron2!=electronCollection->end(); recoElectron2++){

       // Require electron to pass some basic cuts
       if (recoElectron2->et() < 20 && fabs(recoElectron2->eta())<2.5) continue;

       TLorentzVector e1 = TLorentzVector(recoElectron->momentum().x(),recoElectron->momentum().y(),recoElectron->momentum().z(),recoElectron->p());
       TLorentzVector e2 = TLorentzVector(recoElectron2->momentum().x(),recoElectron2->momentum().y(),recoElectron2->momentum().z(),recoElectron2->p());
       TLorentzVector pair=e1+e2;
       float currentInvMass = pair.M();
       if (fabs(currentInvMass-zMass) < fabs(closestEEInvMasstoZFound-zMass)) {
         closestEEInvMasstoZFound = currentInvMass;
	 electron_et  = recoElectron->et();
	 electron_eta = recoElectron->eta();
	 electron_phi = recoElectron->phi();
	 electron2_et  = recoElectron2->et();
         electron2_eta = recoElectron2->eta();
         electron2_phi = recoElectron2->phi();
       }
     }
   }
  }
  ////////////////////////////////////////////////////////////////////////////////



  ////////////////////////////////////////////////////////////////////////////////
  // Take the STA muon container
  Handle<MuonCollection> muonCollection;
  iEvent.getByLabel(theMuonCollectionLabel,muonCollection);

  // put in code for muons here
  ////////////////////////////////////////////////////////////////////////////////  

  

  ////////////////////////////////////////////////////////////////////////////////
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

    // if it overlaps with electron, it is not a jet
    if ( electron_et>0.0 && fabs(i_calojet->eta()-electron_eta ) < 0.2 && calcDeltaPhi(i_calojet->phi(), electron_phi ) < 0.2) continue;
    if ( electron2_et>0.0&& fabs(i_calojet->eta()-electron2_eta) < 0.2 && calcDeltaPhi(i_calojet->phi(), electron2_phi) < 0.2) continue;

    // if it has too low Et, throw away
    if (jet_current_et < 15) continue;

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
  ////////////////////////////////////////////////////////////////////////////////


  ////////////////////////////////////////////////////////////////////////////////
  // Fill histograms if photon & jet found
  if (1) {
    //h_mumu_invMass ->Fill();
    h_ee_invMass->Fill(closestEEInvMasstoZFound);
    h_jet_et   ->Fill(jet_et);
    h_jet_count->Fill(jet_count);
  }
  ////////////////////////////////////////////////////////////////////////////////
}


void EwkDQM::endJob(void) {}


// This always returns only a positive deltaPhi
double EwkDQM::calcDeltaPhi(double phi1, double phi2) {

  double deltaPhi = phi1 - phi2;

  if (deltaPhi < 0) deltaPhi = -deltaPhi;

  if (deltaPhi > 3.1415926) {
    deltaPhi = 2 * 3.1415926 - deltaPhi;
  }

  return deltaPhi;
}
