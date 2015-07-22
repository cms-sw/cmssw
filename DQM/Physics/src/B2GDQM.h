#ifndef B2GDQM_H
#define B2GDQM_H

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

// Trigger stuff
#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/DataKeyTags.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include <DataFormats/EgammaCandidates/interface/GsfElectron.h>

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "DQMServices/Core/interface/MonitorElement.h"

// ParticleFlow
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

// EGamma
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"

// Muon
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/MuonReco/interface/MuonIsolation.h"

// Jets
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

// MET
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"

//
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/Math/interface/deltaPhi.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>

class DQMStore;

class B2GDQM : public DQMEDAnalyzer {

 public:
  B2GDQM(const edm::ParameterSet& ps);
  virtual ~B2GDQM();

 protected:
  virtual void analyze(edm::Event const& e, edm::EventSetup const& eSetup);

  virtual void analyzeJets(edm::Event const& e, edm::EventSetup const& eSetup);
  virtual void analyzeSemiMu(edm::Event const& e,
                             edm::EventSetup const& eSetup);
  virtual void analyzeSemiE(edm::Event const& e, edm::EventSetup const& eSetup);
  virtual void analyzeAllHad(edm::Event const& e,
                             edm::EventSetup const& eSetup);

 private:
  virtual void bookHistograms(DQMStore::IBooker& bei, edm::Run const&,
                              edm::EventSetup const&) override;
  int nLumiSecs_;
  int nEvents_, irun, ievt;

  HLTConfigProvider hltConfigProvider_;
  bool isValidHltConfig_;

  // Variables from config file
  edm::InputTag theTriggerResultsCollection;
  edm::EDGetTokenT<edm::TriggerResults> triggerToken_;

  edm::Handle<edm::TriggerResults> triggerResults_;

  std::vector<edm::InputTag> jetLabels_;
  std::vector<edm::EDGetTokenT<edm::View<reco::Jet> > > jetTokens_;
  edm::InputTag PFMETLabel_;
  edm::EDGetTokenT<std::vector<reco::PFMET> > PFMETToken_;

  edm::InputTag cmsTagLabel_;
  edm::EDGetTokenT<edm::View<reco::BasicJet> > cmsTagToken_;

  edm::InputTag muonLabel_;
  edm::EDGetTokenT<edm::View<reco::Muon> > muonToken_;

  edm::InputTag electronLabel_;
  edm::EDGetTokenT<edm::View<reco::GsfElectron> > electronToken_;

  ///////////////////////////
  // Parameters
  ///////////////////////////

  std::vector<double> jetPtMins_;

  double allHadPtCut_;        // pt of both jets
  double allHadRapidityCut_;  // rapidity difference |y0-y1| max
  double allHadDeltaPhiCut_;  // |phi0 - phi1| min

  double semiMu_HadJetPtCut_;  // min pt of hadronic-side jet
  double semiMu_LepJetPtCut_;  // min pt of leptonic-side jet
  double semiMu_dphiHadCut_;  // min deltaPhi between muon and hadronic-side jet
  double semiMu_dRMin_;  // min deltaR between muon and nearest jet for 2d cut
  double semiMu_ptRel_;  // max ptRel between muon and nearest jet for 2d cut
  std::shared_ptr<StringCutObjectSelector<reco::Muon> >
      muonSelect_;  // Selection on all muons

  double semiE_HadJetPtCut_;  // pt of hadronic-side jet
  double semiE_LepJetPtCut_;  // min pt of leptonic-side jet
  double semiE_dphiHadCut_;   // min deltaPhi between electron and hadronic-side
                              // jet
  double semiE_dRMin_;  // min deltaR between electron and nearest jet for 2d
                        // cut
  double semiE_ptRel_;  // max ptRel between electron and nearest jet for 2d cut
  std::shared_ptr<StringCutObjectSelector<reco::GsfElectron> >
      elecSelect_;  // Kinematic selection on all electrons

  std::string PFJetCorService_;
  ///////////////////////////
  // Histograms
  ///////////////////////////
  std::vector<MonitorElement*> pfJet_pt;
  std::vector<MonitorElement*> pfJet_y;
  std::vector<MonitorElement*> pfJet_phi;
  std::vector<MonitorElement*> pfJet_m;
  std::vector<MonitorElement*> pfJet_chef;
  std::vector<MonitorElement*> pfJet_nhef;
  std::vector<MonitorElement*> pfJet_cemf;
  std::vector<MonitorElement*> pfJet_nemf;
  std::vector<MonitorElement*> boostedJet_subjetPt;
  std::vector<MonitorElement*> boostedJet_subjetY;
  std::vector<MonitorElement*> boostedJet_subjetPhi;
  std::vector<MonitorElement*> boostedJet_subjetM;
  std::vector<MonitorElement*> boostedJet_subjetN;
  std::vector<MonitorElement*> boostedJet_massDrop;
  std::vector<MonitorElement*> boostedJet_minMass;
  MonitorElement* pfMet_pt;
  MonitorElement* pfMet_phi;

  MonitorElement* semiMu_muPt;
  MonitorElement* semiMu_muEta;
  MonitorElement* semiMu_muPhi;
  MonitorElement* semiMu_muDRMin;
  MonitorElement* semiMu_muPtRel;
  MonitorElement* semiMu_hadJetDR;
  MonitorElement* semiMu_hadJetPt;
  MonitorElement* semiMu_hadJetY;
  MonitorElement* semiMu_hadJetPhi;
  MonitorElement* semiMu_hadJetMass;
  MonitorElement* semiMu_hadJetMinMass;
  MonitorElement* semiMu_mttbar;

  MonitorElement* semiE_ePt;
  MonitorElement* semiE_eEta;
  MonitorElement* semiE_ePhi;
  MonitorElement* semiE_eDRMin;
  MonitorElement* semiE_ePtRel;
  MonitorElement* semiE_hadJetDR;
  MonitorElement* semiE_hadJetPt;
  MonitorElement* semiE_hadJetY;
  MonitorElement* semiE_hadJetPhi;
  MonitorElement* semiE_hadJetMass;
  MonitorElement* semiE_hadJetMinMass;
  MonitorElement* semiE_mttbar;

  MonitorElement* allHad_pt0;
  MonitorElement* allHad_y0;
  MonitorElement* allHad_phi0;
  MonitorElement* allHad_mass0;
  MonitorElement* allHad_minMass0;
  MonitorElement* allHad_pt1;
  MonitorElement* allHad_y1;
  MonitorElement* allHad_phi1;
  MonitorElement* allHad_mass1;
  MonitorElement* allHad_minMass1;
  MonitorElement* allHad_mttbar;
};

#endif
