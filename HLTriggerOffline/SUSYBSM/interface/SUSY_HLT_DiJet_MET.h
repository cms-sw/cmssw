#ifndef SUSY_HLT_DiJet_MET_H
#define SUSY_HLT_DiJet_MET_H

// event
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

// DQM
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

// Muon
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

// MET
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"

// Jets
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"

// Trigger
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

class SUSY_HLT_DiJet_MET : public DQMEDAnalyzer {
public:
  SUSY_HLT_DiJet_MET(const edm::ParameterSet &ps);
  ~SUSY_HLT_DiJet_MET() override;

protected:
  void dqmBeginRun(edm::Run const &, edm::EventSetup const &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const &e, edm::EventSetup const &eSetup) override;

private:
  // histos booking function
  void bookHistos(DQMStore::IBooker &);

  // variables from config file
  edm::EDGetTokenT<reco::PFMETCollection> thePfMETCollection_;
  edm::EDGetTokenT<reco::CaloMETCollection> theCaloMETCollection_;
  edm::EDGetTokenT<reco::PFJetCollection> thePfJetCollection_;
  edm::EDGetTokenT<reco::CaloJetCollection> theCaloJetCollection_;
  edm::EDGetTokenT<edm::TriggerResults> triggerResults_;
  edm::EDGetTokenT<trigger::TriggerEvent> theTrigSummary_;

  HLTConfigProvider fHltConfig;

  std::string HLTProcess_;
  std::string triggerPath_;
  std::string triggerPathAuxiliaryForHadronic_;
  edm::InputTag triggerFilter_;
  edm::InputTag triggerJetFilter_;
  double ptThrJetTrig_;
  double etaThrJetTrig_;
  double ptThrJet_;
  double etaThrJet_;
  double metCut_;

  // Histograms
  MonitorElement *h_pfMetPhi;
  MonitorElement *h_pfJetPt;
  MonitorElement *h_pfJetEta;
  MonitorElement *h_pfJetPhi;
  MonitorElement *h_pfJet1Jet2DPhi;
  MonitorElement *h_caloMetvsPFMet;
  MonitorElement *h_triggerMet;
  MonitorElement *h_triggerMetPhi;
  MonitorElement *h_triggerJetPt;
  MonitorElement *h_triggerJetEta;
  MonitorElement *h_triggerJetPhi;
  MonitorElement *h_pfMetTurnOn_num;
  MonitorElement *h_pfMetTurnOn_den;
  MonitorElement *h_pfJet2PtTurnOn_num;
  MonitorElement *h_pfJet2PtTurnOn_den;
};

#endif
