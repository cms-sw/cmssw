#ifndef SUSY_HLT_SingleLepton_H
#define SUSY_HLT_SingleLepton_H

// event
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

// DQM
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

// Electron
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

// Muon
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

// MET
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"

// Jets
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/JetReco/interface/PFJet.h"

// Trigger
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

// Vertices
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

// Conversions
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"

// Beam spot
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

class SUSY_HLT_SingleLepton : public DQMEDAnalyzer {
public:
  SUSY_HLT_SingleLepton(const edm::ParameterSet &ps);
  ~SUSY_HLT_SingleLepton() override;

protected:
  void dqmBeginRun(const edm::Run &run, const edm::EventSetup &e) override;
  void bookHistograms(DQMStore::IBooker &ibooker, const edm::Run &, const edm::EventSetup &) override;
  void analyze(const edm::Event &e, const edm::EventSetup &eSetup) override;

private:
  // variables from config file
  edm::InputTag theElectronTag_;
  edm::EDGetTokenT<reco::GsfElectronCollection> theElectronCollection_;
  edm::InputTag theMuonTag_;
  edm::EDGetTokenT<reco::MuonCollection> theMuonCollection_;
  edm::InputTag thePfMETTag_;
  edm::EDGetTokenT<reco::PFMETCollection> thePfMETCollection_;
  edm::InputTag thePfJetTag_;
  edm::EDGetTokenT<reco::PFJetCollection> thePfJetCollection_;
  edm::InputTag theJetTagTag_;
  edm::EDGetTokenT<reco::JetTagCollection> theJetTagCollection_;

  edm::InputTag theVertexCollectionTag_;
  edm::EDGetTokenT<reco::VertexCollection> theVertexCollection_;
  edm::InputTag theConversionCollectionTag_;
  edm::EDGetTokenT<reco::ConversionCollection> theConversionCollection_;
  edm::InputTag theBeamSpotTag_;
  edm::EDGetTokenT<reco::BeamSpot> theBeamSpot_;

  edm::InputTag theLeptonFilterTag_;
  edm::InputTag theHLTHTTag_;
  edm::EDGetTokenT<reco::METCollection> theHLTHT_;
  edm::InputTag theHLTMETTag_;
  edm::EDGetTokenT<reco::METCollection> theHLTMET_;
  edm::InputTag theHLTJetCollectionTag_;
  edm::EDGetTokenT<reco::CaloJetCollection> theHLTJetCollection_;
  edm::InputTag theHLTJetTagCollectionTag_;
  edm::EDGetTokenT<reco::JetTagCollection> theHLTJetTagCollection_;

  edm::InputTag theTriggerResultsTag_;
  edm::EDGetTokenT<edm::TriggerResults> theTriggerResults_;
  edm::InputTag theTrigSummaryTag_;
  edm::EDGetTokenT<trigger::TriggerEvent> theTrigSummary_;

  HLTConfigProvider fHltConfig_;

  std::string HLTProcess_;

  std::string triggerPath_;
  std::string triggerPathAuxiliary_;
  std::string triggerPathLeptonAuxiliary_;

  double csvlCut_;
  double csvmCut_;
  double csvtCut_;

  double jetPtCut_;
  double jetEtaCut_;
  double metCut_;
  double htCut_;

  double lep_pt_threshold_;
  double ht_threshold_;
  double met_threshold_;
  double csv_threshold_;

  // Histograms
  MonitorElement *h_triggerLepPt_;
  MonitorElement *h_triggerLepEta_;
  MonitorElement *h_triggerLepPhi_;
  MonitorElement *h_HT_;
  MonitorElement *h_MET_;
  MonitorElement *h_maxCSV_;
  MonitorElement *h_leptonTurnOn_num_;
  MonitorElement *h_leptonTurnOn_den_;
  MonitorElement *h_pfHTTurnOn_num_;
  MonitorElement *h_pfHTTurnOn_den_;
  MonitorElement *h_pfMetTurnOn_num_;
  MonitorElement *h_pfMetTurnOn_den_;
  MonitorElement *h_CSVTurnOn_num_;
  MonitorElement *h_CSVTurnOn_den_;
  MonitorElement *h_btagTurnOn_num_;
  MonitorElement *h_btagTurnOn_den_;
};

#endif
