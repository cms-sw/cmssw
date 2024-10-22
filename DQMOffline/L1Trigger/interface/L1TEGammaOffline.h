#ifndef L1TEGammaOffline_H
#define L1TEGammaOffline_H

//Framework
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//event
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

//DQM
#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMOffline/L1Trigger/interface/HistDefinition.h"

//Candidate handling
#include "DataFormats/Candidate/interface/Candidate.h"

// Electron & photon collections
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

// Vertex utilities
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

// Trigger
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

// stage2 collections:
#include "DataFormats/L1Trigger/interface/EGamma.h"

class L1TEGammaOffline : public DQMOneEDAnalyzer<> {
public:
  L1TEGammaOffline(const edm::ParameterSet& ps);
  ~L1TEGammaOffline() override;

  enum PlotConfig { nVertex, ETvsET, PHIvsPHI };

  static const std::map<std::string, unsigned int> PlotConfigNames;

protected:
  void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup) override;
  void dqmEndRun(edm::Run const& run, edm::EventSetup const& eSetup) override;

private:
  bool passesLooseEleId(reco::GsfElectron const& electron) const;
  bool passesMediumEleId(reco::GsfElectron const& electron) const;
  void bookElectronHistos(DQMStore::IBooker&);
  void bookPhotonHistos(DQMStore::IBooker&);

  //other functions
  double Distance(const reco::Candidate& c1, const reco::Candidate& c2);
  double DistancePhi(const reco::Candidate& c1, const reco::Candidate& c2);
  double calcDeltaPhi(double phi1, double phi2);

  void fillElectrons(edm::Event const& e, const unsigned int nVertex);
  void fillPhotons(edm::Event const& e, const unsigned int nVertex);
  bool findTagAndProbePair(edm::Handle<reco::GsfElectronCollection> const& electrons);
  bool matchesAnHLTObject(double eta, double phi) const;

  void normalise2DHistogramsToBinArea();

  math::XYZPoint PVPoint_;

  //variables from config file
  edm::EDGetTokenT<reco::GsfElectronCollection> theGsfElectronCollection_;
  edm::EDGetTokenT<std::vector<reco::Photon> > thePhotonCollection_;
  edm::EDGetTokenT<reco::VertexCollection> thePVCollection_;
  edm::EDGetTokenT<reco::BeamSpot> theBSCollection_;
  edm::EDGetTokenT<trigger::TriggerEvent> triggerInputTag_;
  edm::EDGetTokenT<edm::TriggerResults> triggerResultsInputTag_;
  std::string triggerProcess_;
  std::vector<std::string> triggerNames_;
  std::string histFolder_;
  std::string efficiencyFolder_;

  edm::EDGetTokenT<l1t::EGammaBxCollection> stage2CaloLayer2EGammaToken_;

  std::vector<double> electronEfficiencyThresholds_;
  std::vector<double> electronEfficiencyBins_;
  double probeToL1Offset_;
  std::vector<double> deepInspectionElectronThresholds_;

  std::vector<double> photonEfficiencyThresholds_;
  std::vector<double> photonEfficiencyBins_;

  double maxDeltaRForL1Matching_;
  double maxDeltaRForHLTMatching_;
  double recoToL1TThresholdFactor_;

  reco::GsfElectron tagElectron_;
  reco::GsfElectron probeElectron_;
  double tagAndProbleInvariantMass_;

  HLTConfigProvider hltConfig_;
  std::vector<unsigned int> triggerIndices_;
  edm::TriggerResults triggerResults_;
  trigger::TriggerEvent triggerEvent_;
  dqmoffline::l1t::HistDefinitions histDefinitions_;

  // TODO: add turn-on cuts (vectors of doubles)
  // Histograms
  MonitorElement* h_nVertex_;
  MonitorElement* h_tagAndProbeMass_;

  // electron reco vs L1
  MonitorElement* h_L1EGammaETvsElectronET_EB_;
  MonitorElement* h_L1EGammaETvsElectronET_EE_;
  MonitorElement* h_L1EGammaETvsElectronET_EB_EE_;

  MonitorElement* h_L1EGammaPhivsElectronPhi_EB_;
  MonitorElement* h_L1EGammaPhivsElectronPhi_EE_;
  MonitorElement* h_L1EGammaPhivsElectronPhi_EB_EE_;

  MonitorElement* h_L1EGammaEtavsElectronEta_;

  // electron resolutions
  MonitorElement* h_resolutionElectronET_EB_;
  MonitorElement* h_resolutionElectronET_EE_;
  MonitorElement* h_resolutionElectronET_EB_EE_;

  MonitorElement* h_resolutionElectronPhi_EB_;
  MonitorElement* h_resolutionElectronPhi_EE_;
  MonitorElement* h_resolutionElectronPhi_EB_EE_;

  MonitorElement* h_resolutionElectronEta_;

  // electron turn-ons
  std::map<double, MonitorElement*> h_efficiencyElectronET_EB_pass_;
  std::map<double, MonitorElement*> h_efficiencyElectronET_EE_pass_;
  std::map<double, MonitorElement*> h_efficiencyElectronET_EB_EE_pass_;
  std::map<double, MonitorElement*> h_efficiencyElectronPhi_vs_Eta_pass_;
  // for deep inspection only
  std::map<double, MonitorElement*> h_efficiencyElectronEta_pass_;
  std::map<double, MonitorElement*> h_efficiencyElectronPhi_pass_;
  std::map<double, MonitorElement*> h_efficiencyElectronNVertex_pass_;

  // we could drop the map here, but L1TEfficiency_Harvesting expects
  // identical names except for the suffix
  std::map<double, MonitorElement*> h_efficiencyElectronET_EB_total_;
  std::map<double, MonitorElement*> h_efficiencyElectronET_EE_total_;
  std::map<double, MonitorElement*> h_efficiencyElectronET_EB_EE_total_;
  std::map<double, MonitorElement*> h_efficiencyElectronPhi_vs_Eta_total_;
  // for deep inspection only
  std::map<double, MonitorElement*> h_efficiencyElectronEta_total_;
  std::map<double, MonitorElement*> h_efficiencyElectronPhi_total_;
  std::map<double, MonitorElement*> h_efficiencyElectronNVertex_total_;

  // photons
  MonitorElement* h_L1EGammaETvsPhotonET_EB_;
  MonitorElement* h_L1EGammaETvsPhotonET_EE_;
  MonitorElement* h_L1EGammaETvsPhotonET_EB_EE_;

  MonitorElement* h_L1EGammaPhivsPhotonPhi_EB_;
  MonitorElement* h_L1EGammaPhivsPhotonPhi_EE_;
  MonitorElement* h_L1EGammaPhivsPhotonPhi_EB_EE_;

  MonitorElement* h_L1EGammaEtavsPhotonEta_;

  // electron resolutions
  MonitorElement* h_resolutionPhotonET_EB_;
  MonitorElement* h_resolutionPhotonET_EE_;
  MonitorElement* h_resolutionPhotonET_EB_EE_;

  MonitorElement* h_resolutionPhotonPhi_EB_;
  MonitorElement* h_resolutionPhotonPhi_EE_;
  MonitorElement* h_resolutionPhotonPhi_EB_EE_;

  MonitorElement* h_resolutionPhotonEta_;

  // photon turn-ons
  std::map<double, MonitorElement*> h_efficiencyPhotonET_EB_pass_;
  std::map<double, MonitorElement*> h_efficiencyPhotonET_EE_pass_;
  std::map<double, MonitorElement*> h_efficiencyPhotonET_EB_EE_pass_;

  // we could drop the map here, but L1TEfficiency_Harvesting expects
  // identical names except for the suffix
  std::map<double, MonitorElement*> h_efficiencyPhotonET_EB_total_;
  std::map<double, MonitorElement*> h_efficiencyPhotonET_EE_total_;
  std::map<double, MonitorElement*> h_efficiencyPhotonET_EB_EE_total_;
};

#endif
