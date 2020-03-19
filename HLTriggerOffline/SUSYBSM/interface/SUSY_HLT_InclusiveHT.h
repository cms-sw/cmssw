#ifndef SUSY_HLT_InclusiveHT_H
#define SUSY_HLT_InclusiveHT_H

// event
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

// DQM
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

// MET
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

class SUSY_HLT_InclusiveHT : public DQMEDAnalyzer {
public:
  SUSY_HLT_InclusiveHT(const edm::ParameterSet &ps);
  ~SUSY_HLT_InclusiveHT() override;

protected:
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const &e, edm::EventSetup const &eSetup) override;

private:
  // histos booking function
  void bookHistos(DQMStore::IBooker &);

  // variables from config file
  edm::EDGetTokenT<reco::PFMETCollection> thePfMETCollection_;
  edm::EDGetTokenT<reco::PFJetCollection> thePfJetCollection_;
  edm::EDGetTokenT<reco::CaloJetCollection> theCaloJetCollection_;
  edm::EDGetTokenT<edm::TriggerResults> triggerResults_;
  edm::EDGetTokenT<trigger::TriggerEvent> theTrigSummary_;

  std::string triggerPath_;
  std::string triggerPathAuxiliaryForHadronic_;
  edm::InputTag triggerFilter_;
  double ptThrJet_;
  double etaThrJet_;

  // Histograms
  MonitorElement *h_pfMet;
  MonitorElement *h_pfMetPhi;
  MonitorElement *h_pfHT;
  MonitorElement *h_caloHT;
  MonitorElement *h_pfJetPt;
  MonitorElement *h_pfJetEta;
  MonitorElement *h_pfJetPhi;
  MonitorElement *h_caloJetPt;
  MonitorElement *h_caloJetEta;
  MonitorElement *h_caloJetPhi;
  MonitorElement *h_triggerJetPt;
  MonitorElement *h_triggerJetEta;
  MonitorElement *h_triggerJetPhi;
  MonitorElement *h_triggerMetPt;
  MonitorElement *h_triggerMetPhi;
  MonitorElement *h_triggerHT;
  MonitorElement *h_pfMetTurnOn_num;
  MonitorElement *h_pfMetTurnOn_den;
  MonitorElement *h_pfHTTurnOn_num;
  MonitorElement *h_pfHTTurnOn_den;
};

#endif
