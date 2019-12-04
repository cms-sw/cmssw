#ifndef L1TStage2CaloLayer2Offline_H
#define L1TStage2CaloLayer2Offline_H

//Framework
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//event
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

//DQM
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

//Candidate handling
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

// Electron
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

// PFMET
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"

// Vertex utilities
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

// PFJets
#include "DataFormats/JetReco/interface/PFJet.h"

// Calo MET
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

// Conversions
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"

// Trigger
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "FWCore/Common/interface/TriggerNames.h"

// stage2 collections:
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

#include "DQMOffline/L1Trigger/interface/HistDefinition.h"

class L1TStage2CaloLayer2Offline : public DQMEDAnalyzer {
public:
  L1TStage2CaloLayer2Offline(const edm::ParameterSet& ps);
  ~L1TStage2CaloLayer2Offline() override;

  enum ControlPlots {
    L1MET,
    L1ETMHF,
    L1MHT,
    L1ETT,
    L1HTT,
    OfflineMET,
    OfflineETMHF,
    OfflinePFMetNoMu,
    OfflineMHT,
    OfflineETT,
    OfflineHTT,
    L1JetET,
    OfflineJetET,
    NumberOfControlPlots
  };

  typedef std::map<L1TStage2CaloLayer2Offline::ControlPlots, MonitorElement*> ControlPlotMap;

  enum PlotConfig { nVertex, ETvsET, PHIvsPHI };

  static const std::map<std::string, unsigned int> PlotConfigNames;

protected:
  void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup) override;
  void endJob() override;

private:
  //histos booking function
  void bookHistos(DQMStore::IBooker&);
  void bookEnergySumHistos(DQMStore::IBooker&);
  void bookJetHistos(DQMStore::IBooker&);

  void fillEnergySums(edm::Event const& e, const unsigned int nVertex);
  void fillJets(edm::Event const& e, const unsigned int nVertex);
  void fillJetEfficiencies(const double& recoEt, const double& l1Et, const double& recoEta);

  bool doesNotOverlapWithHLTObjects(const l1t::Jet& jet) const;

  void normalise2DHistogramsToBinArea();

  //private variables
  math::XYZPoint PVPoint_;

  //variables from config file
  edm::EDGetTokenT<reco::PFJetCollection> thePFJetCollection_;
  edm::EDGetTokenT<reco::CaloMETCollection> thecaloMETCollection_;
  edm::EDGetTokenT<reco::CaloMETCollection> thecaloETMHFCollection_;
  edm::EDGetTokenT<reco::PFMETCollection> thePFMETNoMuCollection_;
  edm::EDGetTokenT<reco::VertexCollection> thePVCollection_;
  edm::EDGetTokenT<reco::BeamSpot> theBSCollection_;
  edm::EDGetTokenT<trigger::TriggerEvent> triggerInputTag_;
  edm::EDGetTokenT<edm::TriggerResults> triggerResultsInputTag_;
  std::string triggerProcess_;
  std::vector<std::string> triggerNames_;
  std::string histFolderEtSum_;
  std::string histFolderJet_;
  std::string efficiencyFolderEtSum_;
  std::string efficiencyFolderJet_;

  edm::EDGetTokenT<l1t::JetBxCollection> stage2CaloLayer2JetToken_;
  edm::EDGetTokenT<l1t::EtSumBxCollection> stage2CaloLayer2EtSumToken_;

  std::vector<double> jetEfficiencyThresholds_;
  std::vector<double> metEfficiencyThresholds_;
  std::vector<double> mhtEfficiencyThresholds_;
  std::vector<double> ettEfficiencyThresholds_;
  std::vector<double> httEfficiencyThresholds_;

  std::vector<double> jetEfficiencyBins_;
  std::vector<double> metEfficiencyBins_;
  std::vector<double> mhtEfficiencyBins_;
  std::vector<double> ettEfficiencyBins_;
  std::vector<double> httEfficiencyBins_;

  double recoHTTMaxEta_;
  double recoMHTMaxEta_;

  HLTConfigProvider hltConfig_;
  std::vector<unsigned int> triggerIndices_;
  edm::TriggerResults triggerResults_;
  trigger::TriggerEvent triggerEvent_;
  dqmoffline::l1t::HistDefinitions histDefinitions_;

  // TODO: add turn-on cuts (vectors of doubles)
  // Histograms
  MonitorElement* h_nVertex_;

  // control plots
  ControlPlotMap h_controlPlots_;

  // energy sums reco vs L1
  MonitorElement* h_L1METvsCaloMET_;
  MonitorElement* h_L1ETMHFvsCaloETMHF_;
  MonitorElement* h_L1METvsPFMetNoMu_;
  MonitorElement* h_L1MHTvsRecoMHT_;
  MonitorElement* h_L1METTvsCaloETT_;
  MonitorElement* h_L1HTTvsRecoHTT_;

  MonitorElement* h_L1METPhivsCaloMETPhi_;
  MonitorElement* h_L1ETMHFPhivsCaloETMHFPhi_;
  MonitorElement* h_L1METPhivsPFMetNoMuPhi_;
  MonitorElement* h_L1MHTPhivsRecoMHTPhi_;

  // energy sum resolutions
  MonitorElement* h_resolutionMET_;
  MonitorElement* h_resolutionETMHF_;
  MonitorElement* h_resolutionPFMetNoMu_;
  MonitorElement* h_resolutionMHT_;
  MonitorElement* h_resolutionETT_;
  MonitorElement* h_resolutionHTT_;
  MonitorElement* h_resolutionMETPhi_;
  MonitorElement* h_resolutionETMHFPhi_;
  MonitorElement* h_resolutionPFMetNoMuPhi_;
  MonitorElement* h_resolutionMHTPhi_;

  // energy sum turn ons
  std::map<double, MonitorElement*> h_efficiencyMET_pass_;
  std::map<double, MonitorElement*> h_efficiencyETMHF_pass_;
  std::map<double, MonitorElement*> h_efficiencyPFMetNoMu_pass_;
  std::map<double, MonitorElement*> h_efficiencyMHT_pass_;
  std::map<double, MonitorElement*> h_efficiencyETT_pass_;
  std::map<double, MonitorElement*> h_efficiencyHTT_pass_;

  std::map<double, MonitorElement*> h_efficiencyMET_total_;
  std::map<double, MonitorElement*> h_efficiencyETMHF_total_;
  std::map<double, MonitorElement*> h_efficiencyPFMetNoMu_total_;
  std::map<double, MonitorElement*> h_efficiencyMHT_total_;
  std::map<double, MonitorElement*> h_efficiencyETT_total_;
  std::map<double, MonitorElement*> h_efficiencyHTT_total_;

  // jet reco vs L1
  MonitorElement* h_L1JetETvsPFJetET_HB_;
  MonitorElement* h_L1JetETvsPFJetET_HE_;
  MonitorElement* h_L1JetETvsPFJetET_HF_;
  MonitorElement* h_L1JetETvsPFJetET_HB_HE_;

  MonitorElement* h_L1JetPhivsPFJetPhi_HB_;
  MonitorElement* h_L1JetPhivsPFJetPhi_HE_;
  MonitorElement* h_L1JetPhivsPFJetPhi_HF_;
  MonitorElement* h_L1JetPhivsPFJetPhi_HB_HE_;

  MonitorElement* h_L1JetEtavsPFJetEta_;

  // jet resolutions
  MonitorElement* h_resolutionJetET_HB_;
  MonitorElement* h_resolutionJetET_HE_;
  MonitorElement* h_resolutionJetET_HF_;
  MonitorElement* h_resolutionJetET_HB_HE_;

  MonitorElement* h_resolutionJetPhi_HB_;
  MonitorElement* h_resolutionJetPhi_HE_;
  MonitorElement* h_resolutionJetPhi_HF_;
  MonitorElement* h_resolutionJetPhi_HB_HE_;

  MonitorElement* h_resolutionJetEta_;

  // jet turn-ons
  std::map<double, MonitorElement*> h_efficiencyJetEt_HB_pass_;
  std::map<double, MonitorElement*> h_efficiencyJetEt_HE_pass_;
  std::map<double, MonitorElement*> h_efficiencyJetEt_HF_pass_;
  std::map<double, MonitorElement*> h_efficiencyJetEt_HB_HE_pass_;

  // we could drop the map here, but L1TEfficiency_Harvesting expects
  // identical names except for the suffix
  std::map<double, MonitorElement*> h_efficiencyJetEt_HB_total_;
  std::map<double, MonitorElement*> h_efficiencyJetEt_HE_total_;
  std::map<double, MonitorElement*> h_efficiencyJetEt_HF_total_;
  std::map<double, MonitorElement*> h_efficiencyJetEt_HB_HE_total_;
};

#endif
