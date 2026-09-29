#ifndef DQMOffline_Trigger_interface_MuonDQMVariables_h
#define DQMOffline_Trigger_interface_MuonDQMVariables_h

#include "DQMServices/Components/interface/GenericObjectDQMSource.h"
#include "DataFormats/MuonReco/interface/Muon.h"

template <>
struct DQMVariableTraits<reco::Muon> {
  static std::vector<DQMVariable<reco::Muon>> variables() {
    return {
        {"pt", "p_{T} [GeV]", 100, 0., 200., [](reco::Muon const& m) { return m.pt(); }},
        {"eta", "#eta", 100, -3., 3., [](reco::Muon const& m) { return m.eta(); }},
        {"phi", "#phi", 100, -3.15, 3.15, [](reco::Muon const& m) { return m.phi(); }},
        {"charge", "charge", 5, -2.5, 2.5, [](reco::Muon const& m) { return m.charge(); }},
        {"isGlobalMuon", "isGlobalMuon", 2, -0.5, 1.5, [](reco::Muon const& m) { return m.isGlobalMuon(); }},
        {"isTrackerMuon", "isTrackerMuon", 2, -0.5, 1.5, [](reco::Muon const& m) { return m.isTrackerMuon(); }},
        {"isPFMuon", "isPFMuon", 2, -0.5, 1.5, [](reco::Muon const& m) { return m.isPFMuon(); }},
        {"numberOfMatches", "N_{matched stations}", 10, 0., 10., [](reco::Muon const& m) { return m.numberOfMatches(); }},
        {"pfIsoChargedHadronPt",
         "PF charged had. iso p_{T} [GeV]",
         100,
         0.,
         20.,
         [](reco::Muon const& m) { return m.pfIsolationR04().sumChargedHadronPt; }},
        {"pfIsoNeutralHadronEt",
         "PF neutral had. iso E_{T} [GeV]",
         100,
         0.,
         20.,
         [](reco::Muon const& m) { return m.pfIsolationR04().sumNeutralHadronEt; }},
    };
  }
};

#endif
