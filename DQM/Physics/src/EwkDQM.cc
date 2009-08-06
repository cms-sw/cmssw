/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/07/22 15:00:38 $
 *  $Revision: 1.8 $
 *  \author Michael B. Anderson, University of Wisconsin-Madison
 *  \author Will Parker, University of Wisconsin-Madison
 */

#include "DQM/Physics/src/EwkDQM.h"
#include "DataFormats/Candidate/interface/Candidate.h"
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
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
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
  theElecTriggerPathToPass    = parameters.getParameter<string>("elecTriggerPathToPass");
  theMuonTriggerPathToPass    = parameters.getParameter<string>("muonTriggerPathToPass");
  theTriggerResultsCollection = parameters.getParameter<InputTag>("triggerResultsCollection");
  theMuonCollectionLabel      = parameters.getParameter<InputTag>("muonCollection");
  theElectronCollectionLabel  = parameters.getParameter<InputTag>("electronCollection");
  theCaloJetCollectionLabel   = parameters.getParameter<InputTag>("caloJetCollection");
  theCaloMETCollectionLabel   = parameters.getParameter<InputTag>("caloMETCollection");
}

EwkDQM::~EwkDQM() { 
}


void EwkDQM::beginJob(EventSetup const& iSetup) {

  logTraceName = "EwkAnalyzer";

  LogTrace(logTraceName)<<"Parameters initialization";
  theDbe = Service<DQMStore>().operator->();
  theDbe->setCurrentFolder("Physics/EwkDQM");  // Use folder with name of PAG

  const float pi = 3.14159265;

  // Keep the number of plots and number of bins to a minimum!
  h_mumu_invMass = theDbe->book1D("h_mumu_invMass", "#mu#mu Invariant Mass;InvMass (GeV)"    , 20, 40.0, 140.0);
  h_ee_invMass   = theDbe->book1D("h_ee_invMass",   "ee Invariant Mass;InvMass (Gev)"        , 20, 40.0, 140.0);
  h_jet_et       = theDbe->book1D("h_jet_et",       "Jet with highest E_{T} (from "+theCaloJetCollectionLabel.label()+");E_{T}(1^{st} jet) (GeV)",    20, 0., 200.0);
  h_jet2_et      = theDbe->book1D("h_jet2_et",      "Jet with 2^{nd} highest E_{T} (from "+theCaloJetCollectionLabel.label()+");E_{T}(2^{nd} jet) (GeV)",    20, 0., 200.0);
  h_jet_count    = theDbe->book1D("h_jet_count",    "Number of "+theCaloJetCollectionLabel.label()+" (E_{T} > 15 GeV);Number of Jets", 8, -0.5, 7.5);
  h_e1_et        = theDbe->book1D("h_e1_et",  "E_{T} of Leading Electron;E_{T} (GeV)"        , 20,  0.0 , 100.0);
  h_e2_et        = theDbe->book1D("h_e2_et",  "E_{T} of Second Electron;E_{T} (GeV)"         , 20,  0.0 , 100.0);
  h_e1_eta       = theDbe->book1D("h_e1_eta", "#eta of Leading Electron;#eta"                , 20, -4.0 , 4.0);
  h_e2_eta       = theDbe->book1D("h_e2_eta", "#eta of Second Electron;#eta"                 , 20, -4.0 , 4.0);
  h_e1_phi       = theDbe->book1D("h_e1_phi", "#phi of Leading Electron;#phi"                , 22, (-1.-1./10.)*pi, (1.+1./10.)*pi );
  h_e2_phi       = theDbe->book1D("h_e2_phi", "#phi of Second Electron;#phi"                 , 22, (-1.-1./10.)*pi, (1.+1./10.)*pi );
  h_m1_pt        = theDbe->book1D("h_m1_pt",  "p_{T} of Leading Muon;p_{T}(1^{st} #mu) (GeV)", 20,  0.0 , 100.0);
  h_m2_pt        = theDbe->book1D("h_m2_pt",  "p_{T} of Second Muon;p_{T}(2^{nd} #mu) (GeV)" , 20,  0.0 , 100.0);
  h_m1_eta       = theDbe->book1D("h_m1_eta", "#eta of Leading Muon;#eta(1^{st} #mu)"        , 20, -4.0 , 4.0);
  h_m2_eta       = theDbe->book1D("h_m2_eta", "#eta of Second Muon;#eta(2^{nd} #mu)"         , 20, -4.0 , 4.0);
  h_m1_phi       = theDbe->book1D("h_m1_phi", "#phi of Leading Muon;#phi(1^{st} #mu)"        , 20, (-1.-1./10.)*pi, (1.+1./10.)*pi);
  h_m2_phi       = theDbe->book1D("h_m2_phi", "#phi of Second Muon;#phi(2^{nd} #mu)"         , 20, (-1.-1./10.)*pi, (1.+1./10.)*pi);
//  h_t1_et          = theDbe->book1D("h_t1_et",           "E_{T} of Leading Tau;E_{T} (GeV)" , 20, 0.0 , 100.0);
//  h_t1_eta         = theDbe->book1D("h_t1_eta",          "#eta of Leading Tau;#eta"               , 20, -4.0, 4.0);
//  h_t1_phi         = theDbe->book1D("h_t1_phi",          "#phi of Leading Tau;#phi"               , 20, -4.0, 4.0);
  h_met          = theDbe->book1D("h_met",        "Missing E_{T}; GeV"                       , 20,  0.0 , 100);
  h_met_phi      = theDbe->book1D("h_met_phi",    "Missing E_{T} #phi;#phi(MET)"             , 22, (-1.-1./10.)*pi, (1.+1./10.)*pi );
  h_e_invWMass   = theDbe->book1D("h_e_invWMass", "W-> e #nu Transverse Mass;M_{T} (GeV)"    , 20,  0.0, 140.0); 
  h_m_invWMass   = theDbe->book1D("h_m_invWMass", "W-> #mu #nu Transverse Mass;M_{T} (GeV)"  , 20,  0.0, 140.0); 
}


void EwkDQM::analyze(const Event& iEvent, const EventSetup& iSetup) {

  LogTrace(logTraceName)<<"Analysis of event # ";
  // Did it pass certain HLT path?
  Handle<TriggerResults> HLTresults;
  iEvent.getByLabel(theTriggerResultsCollection, HLTresults); 
  if ( !HLTresults.isValid() ) return;
  HLTConfigProvider hltConfig;
  hltConfig.init("HLT8E29");
  unsigned int triggerIndex_elec = hltConfig.triggerIndex(theElecTriggerPathToPass);
  unsigned int triggerIndex_muon = hltConfig.triggerIndex(theMuonTriggerPathToPass);
  bool passed_electron_HLT = false;
  bool passed_muon_HLT     = false;
  if (triggerIndex_elec < HLTresults->size()) passed_electron_HLT = HLTresults->accept(triggerIndex_elec);
  if (triggerIndex_muon < HLTresults->size()) passed_muon_HLT     = HLTresults->accept(triggerIndex_muon);
  if ( !(passed_electron_HLT || passed_muon_HLT) ) return;


  ////////////////////////////////////////////////////////////////////////////////
  //Missing ET
  Handle<CaloMETCollection> caloMETCollection;
  iEvent.getByLabel(theCaloMETCollectionLabel, caloMETCollection);
  if ( !caloMETCollection.isValid() ) return;
  float missing_et = caloMETCollection->begin()->et();
  float met_phi = caloMETCollection->begin()->phi();


  ////////////////////////////////////////////////////////////////////////////////
  // grab "gaussian sum fitting" electrons
  Handle<GsfElectronCollection> electronCollection;
  iEvent.getByLabel(theElectronCollectionLabel, electronCollection);
  if ( !electronCollection.isValid() ) return;

  // Find the highest and 2nd highest electron
  float electron_et   = -8.0;
  float electron_eta  = -8.0;
  float electron_phi  = -8.0;
  float electron2_et  = -9.0;
  float electron2_eta = -9.0;
  float electron2_phi = -9.0;
  float ee_invMass    = -9.0;
  TLorentzVector e1, e2;

  // If it passed electron HLT and the collection was found, find electrons near Z mass
  if( passed_electron_HLT ) {

    for (reco::GsfElectronCollection::const_iterator recoElectron=electronCollection->begin(); recoElectron!=electronCollection->end(); recoElectron++){

      // Require electron to pass some basic cuts
      if ( recoElectron->et() < 20 || fabs(recoElectron->eta())>2.5 ) continue;

      // Tighter electron cuts
      if ( recoElectron->deltaPhiSuperClusterTrackAtVtx() > 0.58 || 
           recoElectron->deltaEtaSuperClusterTrackAtVtx() > 0.01 || 
           recoElectron->sigmaIetaIeta() > 0.027 ) continue;

      if (recoElectron->et() > electron_et){
	electron2_et  = electron_et;  // 2nd highest gets values from current highest
	electron2_eta = electron_eta;
        electron2_phi = electron_phi;
        electron_et   = recoElectron->et();  // 1st highest gets values from new highest
        electron_eta  = recoElectron->eta();
        electron_phi  = recoElectron->phi();
        e1 = TLorentzVector(recoElectron->momentum().x(),recoElectron->momentum().y(),recoElectron->momentum().z(),recoElectron->p());
      } else if (recoElectron->et() > electron2_et) {
	electron2_et  = recoElectron->et();
        electron2_eta = recoElectron->eta();
        electron2_phi = recoElectron->phi();
	e2 = TLorentzVector(recoElectron->momentum().x(),recoElectron->momentum().y(),recoElectron->momentum().z(),recoElectron->p());
      }
    } // end of loop over electrons
    if (electron2_et>0.0) {
      TLorentzVector pair=e1+e2;
      ee_invMass = pair.M();
    }
  } // end of "are electrons valid"
  ////////////////////////////////////////////////////////////////////////////////



  ////////////////////////////////////////////////////////////////////////////////
  // Take the STA muon container
  Handle<MuonCollection> muonCollection;
  iEvent.getByLabel(theMuonCollectionLabel,muonCollection);
  if ( !muonCollection.isValid() ) return;

  // Find the highest pt muons
  float mm_invMass = -9.0;
  float muon_pt   = -9.0;
  float muon_eta  = -9.0;
  float muon_phi  = -9.0;
  float muon2_pt  = -9.0;
  float muon2_eta = -9.0;
  float muon2_phi = -9.0;
  TLorentzVector m1, m2;

  if( passed_muon_HLT ) {
    for (reco::MuonCollection::const_iterator recoMuon=muonCollection->begin(); recoMuon!=muonCollection->end(); recoMuon++){

      // Require muon to pass some basic cuts
      if ( recoMuon->pt() < 20 || !recoMuon->isGlobalMuon() ) continue;
      // Some tighter muon cuts
      if ( recoMuon->globalTrack()->normalizedChi2() > 10 ) continue;

      if (recoMuon->pt() > muon_pt){
        muon2_pt  = muon_pt;  // 2nd highest gets values from current highest    
        muon2_eta = muon_eta;
        muon2_phi = muon_phi;
        muon_pt   = recoMuon->pt();  // 1st highest gets values from new highest
        muon_eta  = recoMuon->eta();
        muon_phi  = recoMuon->phi();
        m1 = TLorentzVector(recoMuon->momentum().x(),recoMuon->momentum().y(),recoMuon->momentum().z(),recoMuon->p());
      } else if (recoMuon->pt() > muon2_pt) {
        muon2_pt  = recoMuon->pt();
        muon2_eta = recoMuon->eta();
        muon2_phi = recoMuon->phi();
	m2 = TLorentzVector(recoMuon->momentum().x(),recoMuon->momentum().y(),recoMuon->momentum().z(),recoMuon->p());
      }
    }
  }
  if (muon2_pt>0.0) {
    TLorentzVector pair=m1+m2;
    mm_invMass = pair.M();
  }
  ////////////////////////////////////////////////////////////////////////////////  


  ////////////////////////////////////////////////////////////////////////////////
  // Find the highest et jet
  Handle<CaloJetCollection> caloJetCollection;
  iEvent.getByLabel (theCaloJetCollectionLabel,caloJetCollection);
  if ( !caloJetCollection.isValid() ) return;

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
  //                 Fill Histograms                                            //
  ////////////////////////////////////////////////////////////////////////////////

  bool fill_e1  = false;
  bool fill_e2  = false;
  bool fill_m1  = false;
  bool fill_m2  = false;
  bool fill_met = false;

  // Was Z->ee found?
  if (ee_invMass>0.0) {
    h_ee_invMass ->Fill(ee_invMass);
    fill_e1 = true;
    fill_e2 = true;
  }

  // Was Z->mu mu found?
  if (mm_invMass > 0.0) {
    h_mumu_invMass->Fill(mm_invMass);
    fill_m1 = true;
    fill_m2 = true;
  }

  // Was W->e nu found?
  if (electron_et>0.0&&missing_et>20.0) {
    float dphiW = fabs(met_phi-electron_phi);
    float W_mt_e = sqrt(2*missing_et*electron_et*(1-cos(dphiW)));
    h_e_invWMass ->Fill(W_mt_e);
    fill_e1  = true;
    fill_met = true;
  }

  // Was W->mu nu found?
  if (muon_pt>0.0&&missing_et>20.0) {
    float dphiW = fabs(met_phi-muon_phi);
    float W_mt_m = sqrt(2*missing_et*muon_pt*(1-cos(dphiW)));
    h_m_invWMass ->Fill(W_mt_m);
    fill_m1  = true;
    fill_met = true;
  }

  if (jet_et>0.0) {
    h_jet_et   ->Fill(jet_et);
    h_jet_count->Fill(jet_count);
  }

  if (fill_e1) {
    h_e1_et      ->Fill(electron_et);
    h_e1_eta     ->Fill(electron_eta);
    h_e1_phi     ->Fill(electron_phi);
  }
  if (fill_e2) {
    h_e2_et      ->Fill(electron2_et);
    h_e2_eta     ->Fill(electron2_eta);
    h_e2_phi     ->Fill(electron2_phi);
  }
  if (fill_m1) {
    h_m1_pt      ->Fill(muon_pt);
    h_m1_eta     ->Fill(muon_eta);
    h_m1_phi     ->Fill(muon_phi);
  }
  if (fill_m2) {
    h_m2_pt      ->Fill(muon2_pt);
    h_m2_eta     ->Fill(muon2_eta);
    h_m2_phi     ->Fill(muon2_phi);
  }
  if (fill_met) {
    h_met        ->Fill(missing_et);
    h_met_phi    ->Fill(met_phi);
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
