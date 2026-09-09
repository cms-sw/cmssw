#ifndef DQMOffline_Trigger_interface_TrackDQMVariables_h
#define DQMOffline_Trigger_interface_TrackDQMVariables_h

#include "DQMServices/Components/interface/GenericObjectDQMSource.h"
#include "DataFormats/TrackReco/interface/Track.h"

template <>
struct DQMVariableTraits<reco::Track> {
  static std::vector<DQMVariable<reco::Track>> variables() {
    return {
        {"pt", "p_{T} [GeV]", 100, 0., 200., [](reco::Track const& t) { return t.pt(); }},
        {"eta", "#eta", 100, -3., 3., [](reco::Track const& t) { return t.eta(); }},
        {"phi", "#phi", 100, -3.15, 3.15, [](reco::Track const& t) { return t.phi(); }},
        {"chi2", "#chi^{2}", 100, 0., 100., [](reco::Track const& t) { return t.chi2(); }},
        {"ndof", "ndof", 50, 0., 50., [](reco::Track const& t) { return t.ndof(); }},
        {"normalizedChi2", "#chi^{2}/ndof", 100, 0., 10., [](reco::Track const& t) { return t.normalizedChi2(); }},
        {"numberOfValidHits", "N_{valid hits}", 40, 0., 40., [](reco::Track const& t) { return t.numberOfValidHits(); }},
        {"numberOfLostHits", "N_{lost hits}", 20, 0., 20., [](reco::Track const& t) { return t.numberOfLostHits(); }},
        {"dxy", "d_{xy} [cm]", 100, -1., 1., [](reco::Track const& t) { return t.dxy(); }},
        {"dz", "d_{z} [cm]", 100, -20., 20., [](reco::Track const& t) { return t.dz(); }},
        {"charge", "charge", 5, -2.5, 2.5, [](reco::Track const& t) { return t.charge(); }},
        {"qoverp", "q/p [1/GeV]", 100, -1., 1., [](reco::Track const& t) { return t.qoverp(); }},
    };
  }
};

#endif
