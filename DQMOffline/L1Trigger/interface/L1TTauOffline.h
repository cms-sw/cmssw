#ifndef L1TTauOffline_H
#define L1TTauOffline_H

#include "DataFormats/L1Trigger/interface/Muon.h"		 		
#include "DataFormats/L1Trigger/interface/BXVector.h"	
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TrackTransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "TRegexp.h"
#include "TString.h"

#include <iostream>
#include <fstream>
#include <utility>
#include <vector>

//Framework
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//event
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

//DQM
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//Candidate handling
#include "DataFormats/Candidate/interface/Candidate.h"

// Electron & photon collections
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "DataFormats/TauReco/interface/PFTau.h"

// Vertex utilities
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

// Trigger
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

// stage2 collections:
#include "DataFormats/L1Trigger/interface/Tau.h"


class TauL1TPair {

 public :

  TauL1TPair(const reco::PFTau *tau, const l1t::Tau *regTau) : 				
  m_tau(tau), m_regTau(regTau), m_eta(999.), m_phi_bar(999.), m_phi_end(999.) { };
    
  TauL1TPair(const TauL1TPair& tauL1tPair);

  ~TauL1TPair() { };

  double dR();
  double eta() const { return m_tau->eta(); };
  double phi() const { return m_tau->phi(); };
  double pt()  const { return m_tau->pt(); };

  double l1tPt() const { return m_regTau ? m_regTau->pt() : -1.; };					
  double l1tIso() const { return m_regTau ? m_regTau->hwIso() : -1.; };
  double l1tPhi() const { return m_regTau ? m_regTau->phi() : -5.; };					
  double l1tEta() const { return m_regTau ? m_regTau->eta() : -5.; };					

private :

  const reco::PFTau *m_tau;
  const l1t::Tau *m_regTau;					

  double m_eta;
  double m_phi_bar;
  double m_phi_end;

};

class L1TTauOffline: public DQMEDAnalyzer {

public:

  L1TTauOffline(const edm::ParameterSet& ps);
  virtual ~L1TTauOffline();

protected:

  void dqmBeginRun(const edm::Run& run, const edm::EventSetup& iSetup) override;
  /* void dqmBeginRun(edm::Run const &, edm::EventSetup const &) override; */
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup);
  void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& eSetup);
  void endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& eSetup);
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup);

  const reco::Vertex getPrimaryVertex( edm::Handle<reco::VertexCollection> & vertex, edm::Handle<reco::BeamSpot> & beamSpot );
  bool matchHlt(edm::Handle<trigger::TriggerEvent>  & triggerEvent, const reco::Muon * muon);

  // Cut and Matching
  void getTauL1tPairs(edm::Handle<l1t::TauBxCollection> & l1tCands);	
  void getTightMuons(edm::Handle<reco::MuonCollection> & muons, edm::Handle<reco::PFMETCollection> &mets, const reco::Vertex & vertex, edm::Handle<trigger::TriggerEvent> & trigEvent);
  void getProbeTaus(const edm::Event& e, edm::Handle<reco::PFTauCollection> & taus, edm::Handle<reco::MuonCollection> & muons, const reco::Vertex & vertex);

private:
  void bookTauHistos(DQMStore::IBooker &);
  /* void bookPhotonHistos(DQMStore::IBooker &); */

  //other functions
  double Distance(const reco::Candidate & c1, const reco::Candidate & c2);
  double DistancePhi(const reco::Candidate & c1, const reco::Candidate & c2);
  double calcDeltaPhi(double phi1, double phi2);

  math::XYZPoint PVPoint_;

  HLTConfigProvider m_hltConfig;

  edm::ESHandle<MagneticField> m_BField;
  edm::ESHandle<Propagator> m_propagatorAlong;
  edm::ESHandle<Propagator> m_propagatorOpposite;

  //variables from config file
  edm::EDGetTokenT<reco::PFTauCollection> theTauCollection_;
  edm::EDGetTokenT<reco::PFTauDiscriminator> AntiMuInputTag_;
  edm::EDGetTokenT<reco::PFTauDiscriminator> AntiEleInputTag_;
  edm::EDGetTokenT<reco::PFTauDiscriminator> DecayModeFindingInputTag_;
  edm::EDGetTokenT<reco::PFTauDiscriminator> comb3TInputTag_;
  edm::EDGetTokenT<reco::MuonCollection> MuonInputTag_;
  edm::EDGetTokenT<reco::PFMETCollection> MetInputTag_;
  edm::EDGetTokenT<reco::VertexCollection> VtxInputTag_;
  edm::EDGetTokenT<reco::BeamSpot> BsInputTag_;
  edm::EDGetTokenT<trigger::TriggerEvent> triggerEvent_;
  std::string trigProcess_;
  edm::EDGetTokenT<edm::TriggerResults> triggerResults_;
  std::vector<std::string> triggerPath_;
  std::string histFolder_;
  std::string efficiencyFolder_;
  edm::EDGetTokenT<l1t::TauBxCollection> stage2CaloLayer2TauToken_;
  std::vector<double> tauEfficiencyThresholds_;
  std::vector<double> tauEfficiencyBins_; 

  std::vector<const reco::Muon*>  m_TightMuons;
  std::vector<const reco::PFTau*>  m_ProbeTaus;
  std::vector<TauL1TPair>  m_TauL1tPairs;


  std::vector<reco::PFTauCollection>  m_RecoTaus;		
  std::vector<l1t::TauBxCollection>  m_L1tTaus;		  
  std::vector<reco::PFTau>  m_RecoRecoTaus;		
  BXVector<l1t::Tau>  m_L1tL1tTaus;		  
  
  // config params
  std::vector<int> m_L1tPtCuts;

  float m_MaxTauEta;
  float m_MaxL1tTauDR;
  float m_MaxHltTauDR;

  std::vector<int> m_trigIndices;

  // Histograms
  MonitorElement* h_nVertex_;
  MonitorElement* h_tagAndProbeMass_;

  // electron reco vs L1
  MonitorElement* h_L1TauETvsTauET_EB_;
  MonitorElement* h_L1TauETvsTauET_EE_;
  MonitorElement* h_L1TauETvsTauET_EB_EE_;

  MonitorElement* h_L1TauPhivsTauPhi_EB_;
  MonitorElement* h_L1TauPhivsTauPhi_EE_;
  MonitorElement* h_L1TauPhivsTauPhi_EB_EE_;

  MonitorElement* h_L1TauEtavsTauEta_;

  // electron resolutions
  MonitorElement* h_resolutionTauET_EB_;
  MonitorElement* h_resolutionTauET_EE_;
  MonitorElement* h_resolutionTauET_EB_EE_;

  MonitorElement* h_resolutionTauPhi_EB_;
  MonitorElement* h_resolutionTauPhi_EE_;
  MonitorElement* h_resolutionTauPhi_EB_EE_;

  MonitorElement* h_resolutionTauEta_;

  // tau turn-ons
  std::map<double, MonitorElement*> h_efficiencyIsoTauET_EB_pass_;
  std::map<double, MonitorElement*> h_efficiencyIsoTauET_EE_pass_;
  std::map<double, MonitorElement*> h_efficiencyIsoTauET_EB_EE_pass_;

  std::map<double, MonitorElement*> h_efficiencyNonIsoTauET_EB_pass_;
  std::map<double, MonitorElement*> h_efficiencyNonIsoTauET_EE_pass_;
  std::map<double, MonitorElement*> h_efficiencyNonIsoTauET_EB_EE_pass_;

  // we could drop the map here, but L1TEfficiency_Harvesting expects
  // identical names except for the suffix
  std::map<double, MonitorElement*> h_efficiencyIsoTauET_EB_total_;
  std::map<double, MonitorElement*> h_efficiencyIsoTauET_EE_total_;
  std::map<double, MonitorElement*> h_efficiencyIsoTauET_EB_EE_total_;

  std::map<double, MonitorElement*> h_efficiencyNonIsoTauET_EB_total_;
  std::map<double, MonitorElement*> h_efficiencyNonIsoTauET_EE_total_;
  std::map<double, MonitorElement*> h_efficiencyNonIsoTauET_EB_EE_total_;

};

#endif
