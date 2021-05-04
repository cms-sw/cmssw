#ifndef SUSY_HLT_alphaT_H
#define SUSY_HLT_alphaT_H

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

#include "HLTrigger/JetMET/interface/AlphaT.h"

class SUSY_HLT_alphaT : public DQMEDAnalyzer {
public:
  SUSY_HLT_alphaT(const edm::ParameterSet &ps);
  ~SUSY_HLT_alphaT() override;

protected:
  void dqmBeginRun(edm::Run const &, edm::EventSetup const &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const &e, edm::EventSetup const &eSetup) override;

private:
  // histos booking function
  void bookHistos(DQMStore::IBooker &);

  // variables from config file
  edm::EDGetTokenT<reco::PFJetCollection> thePfJetCollection_;
  // edm::EDGetTokenT<reco::CaloJetCollection> theCaloJetCollection_;
  edm::EDGetTokenT<edm::TriggerResults> triggerResults_;
  edm::EDGetTokenT<trigger::TriggerEvent> theTrigSummary_;

  HLTConfigProvider fHltConfig;

  std::string HLTProcess_;
  std::string triggerPath_;
  std::string triggerPathAuxiliaryForMuon_;
  std::string triggerPathAuxiliaryForHadronic_;
  edm::InputTag triggerPreFilter_;
  edm::InputTag triggerFilter_;
  double ptThrJet_;
  double etaThrJet_;
  double pfAlphaTThrTurnon_;
  double pfHtThrTurnon_;
  /* double caloAlphaTThrTurnon_; */
  /* double caloHtThrTurnon_; */

  // Histograms
  /* MonitorElement* h_triggerCaloHt; */
  /* MonitorElement* h_triggerCaloAlphaT; */
  /* MonitorElement* h_triggerCaloAlphaT_triggerCaloHt; */
  /* MonitorElement* h_caloAlphaTTurnOn_num; */
  /* MonitorElement* h_caloAlphaTTurnOn_den; */
  /* MonitorElement* h_caloHtTurnOn_num; */
  /* MonitorElement* h_caloHtTurnOn_den; */

  MonitorElement *h_triggerPfHt;
  MonitorElement *h_triggerPfAlphaT;
  MonitorElement *h_triggerPfAlphaT_triggerPfHt;
  MonitorElement *h_pfAlphaTTurnOn_num;
  MonitorElement *h_pfAlphaTTurnOn_den;
  MonitorElement *h_pfHtTurnOn_num;
  MonitorElement *h_pfHtTurnOn_den;
};

#endif
