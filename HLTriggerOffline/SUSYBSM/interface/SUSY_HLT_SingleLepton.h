#ifndef SUSY_HLT_SingleLepton_H
#define SUSY_HLT_SingleLepton_H

//event
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

//DQM
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//Electron
//#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

//Muon
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

// MET
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"

// Jets
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/BTauReco/interface/JetTag.h"

// Trigger
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

class SUSY_HLT_SingleLepton: public DQMEDAnalyzer{

 public:
  SUSY_HLT_SingleLepton(const edm::ParameterSet& ps);
  virtual ~SUSY_HLT_SingleLepton();

 protected:
  void dqmBeginRun(const edm::Run &run, const edm::EventSetup &e) override;
  void bookHistograms(DQMStore::IBooker &ibooker, const edm::Run &, const edm::EventSetup &) override;
  void beginLuminosityBlock(const edm::LuminosityBlock &lumi, const edm::EventSetup &eSetup) ;
  void analyze(const edm::Event &e, const edm::EventSetup &eSetup);
  void endLuminosityBlock(const edm::LuminosityBlock &lumi, const edm::EventSetup &eSetup);
  void endRun(const edm::Run &run, const edm::EventSetup &eSetup);

 private:
  //variables from config file
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
  edm::InputTag theTriggerResultsTag_;
  edm::EDGetTokenT<edm::TriggerResults> theTriggerResults_;
  edm::InputTag theTrigSummaryTag_;
  edm::EDGetTokenT<trigger::TriggerEvent> theTrigSummary_;

  HLTConfigProvider fHltConfig_;

  std::string HLTProcess_;
  std::string triggerPath_;
  std::string triggerPathAuxiliary_;
  edm::InputTag triggerFilter_;
  double jetPtCut_;
  double jetEtaCut_;
  double leptonPtCut_;
  double leptonPtPlateau_;
  double htPlateau_;
  double metPlateau_;
  double csvPlateau_;
  
  // Histograms
  MonitorElement* h_triggerLepPt_;
  MonitorElement* h_triggerLepEta_;
  MonitorElement* h_triggerLepPhi_;
  MonitorElement* h_CSVTurnOn_num_;
  MonitorElement* h_CSVTurnOn_den_;
  MonitorElement* h_pfMetTurnOn_num_;
  MonitorElement* h_pfMetTurnOn_den_;
  MonitorElement* h_pfHTTurnOn_num_;
  MonitorElement* h_pfHTTurnOn_den_;
  MonitorElement* h_leptonTurnOn_num_;
  MonitorElement* h_leptonTurnOn_den_;
};

#endif
