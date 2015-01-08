#ifndef EwkDQM_H
#define EwkDQM_H

/** \class EwkDQM
 *
 *  DQM offline for SMP V+Jets
 *
 *  \author Valentina Gori, University of Firenze
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"

// Trigger stuff
#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

namespace reco {
class Jet;
class MET;
}
class DQMStore;
class MonitorElement;

class EwkDQM : public DQMEDAnalyzer {
 public:
  /// Constructor
  EwkDQM(const edm::ParameterSet&);

  /// Destructor
  virtual ~EwkDQM();

  ///
  //Book histograms
  void bookHistograms(DQMStore::IBooker &,
    edm::Run const &, edm::EventSetup const &) override;
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&);

  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&);

  double calcDeltaPhi(double phi1, double phi2);

 private:
  // ----------member data ---------------------------

  // Switch for verbosity
  std::string logTraceName;

  HLTConfigProvider hltConfigProvider_;
  bool isValidHltConfig_;

  // Variables from config file
  std::vector<std::string> theElecTriggerPathToPass_;
  std::vector<std::string> theMuonTriggerPathToPass_;
  edm::InputTag thePFJetCollectionLabel_;
  edm::InputTag theCaloMETCollectionLabel_;
  edm::InputTag theTriggerResultsCollection_;
  edm::EDGetTokenT<edm::TriggerResults> theTriggerResultsToken_;
  edm::EDGetTokenT<reco::MuonCollection> theMuonCollectionLabel_;
  edm::EDGetTokenT<reco::GsfElectronCollection> theElectronCollectionLabel_;
  edm::EDGetTokenT<edm::View<reco::Jet> > thePFJetCollectionToken_;
  edm::EDGetTokenT<edm::View<reco::MET> > theCaloMETCollectionToken_;
  edm::EDGetTokenT<reco::VertexCollection> theVertexToken_;

  double eJetMin_;

  // Histograms
  MonitorElement* h_vertex_number;
  MonitorElement* h_vertex_chi2;
  MonitorElement* h_vertex_numTrks;
  MonitorElement* h_vertex_sumTrks;
  MonitorElement* h_vertex_d0;

  MonitorElement* h_jet_count;
  MonitorElement* h_jet_et;
  MonitorElement* h_jet_pt;
  MonitorElement* h_jet_eta;
  MonitorElement* h_jet_phi;

  MonitorElement* h_jet2_et;
  // MonitorElement* h_jet2_pt;
  MonitorElement* h_jet2_eta;
  MonitorElement* h_jet2_phi;

  MonitorElement* h_e1_et;
  MonitorElement* h_e2_et;
  MonitorElement* h_e1_eta;
  MonitorElement* h_e2_eta;
  MonitorElement* h_e1_phi;
  MonitorElement* h_e2_phi;

  MonitorElement* h_m1_pt;
  MonitorElement* h_m2_pt;
  MonitorElement* h_m1_eta;
  MonitorElement* h_m2_eta;
  MonitorElement* h_m1_phi;
  MonitorElement* h_m2_phi;

  // MonitorElement* h_t1_et;
  // MonitorElement* h_t1_eta;
  // MonitorElement* h_t1_phi;

  MonitorElement* h_met;
  MonitorElement* h_met_phi;

  MonitorElement* h_e_invWMass;
  MonitorElement* h_m_invWMass;
  MonitorElement* h_mumu_invMass;
  MonitorElement* h_ee_invMass;
};
#endif

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
