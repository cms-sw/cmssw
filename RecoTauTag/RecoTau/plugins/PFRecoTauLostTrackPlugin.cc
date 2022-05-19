/*
 * =============================================================================
 *       Filename:  PFRecoTauLostTrackPlugin.cc
 *
 *    Description: Add references to tracks of tau-charged-hadrons built on 
 *                 top of a track
 *
 *        Created:  25/04/2022
 *
 *         Authors:  Michal Bluj (NCBJ, Warsaw)
 *
 * =============================================================================
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadron.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadronFwd.h"

namespace reco {
  namespace tau {

    class PFRecoTauLostTrackPlugin : public RecoTauModifierPlugin {
    public:
      explicit PFRecoTauLostTrackPlugin(const edm::ParameterSet&, edm::ConsumesCollector&& iC);
      ~PFRecoTauLostTrackPlugin() override = default;
      void operator()(PFTau&) const override;
      void beginEvent() override;
      void endEvent() override;

    private:
      edm::Handle<reco::TrackCollection> tracks_;
      const edm::EDGetTokenT<reco::TrackCollection> track_token_;
      const int verbosity_;
    };

    PFRecoTauLostTrackPlugin::PFRecoTauLostTrackPlugin(const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC)
        : RecoTauModifierPlugin(cfg, std::move(iC)),
          track_token_(iC.consumes(cfg.getParameter<edm::InputTag>("trackSrc"))),
          verbosity_(cfg.getParameter<int>("verbosity")) {}

    void PFRecoTauLostTrackPlugin::beginEvent() { evt()->getByToken(track_token_, tracks_); }

    void PFRecoTauLostTrackPlugin::operator()(PFTau& tau) const {
      if (!tracks_.isValid()) {  //track collection not available in the event
        if (verbosity_) {
          edm::LogPrint("<PFRecoTauLostTrackPlugin::operator()>:")
              << " Track collection " << tracks_.provenance() << " is not valid."
              << " No tracks will be added to tau.";
        }
        return;
      }
      reco::TrackRefVector lostTracks;
      const PFRecoTauChargedHadronCollection& chargedHadrons = tau.signalTauChargedHadronCandidates();
      for (const auto& chargedHadron : chargedHadrons) {
        if (chargedHadron.algoIs(PFRecoTauChargedHadron::kTrack) && chargedHadron.getTrack().isNonnull()) {
          reco::TrackRef trackRef(tracks_, chargedHadron.getTrack().key());
          lostTracks.push_back(trackRef);
        }
      }
      if (verbosity_) {
        edm::LogPrint("<PFRecoTauLostTrackPlugin::operator()>:")
            << " tau: Pt = " << tau.pt() << ", eta = " << tau.eta() << ", phi = " << tau.phi()
            << ", mass = " << tau.mass() << " (decayMode = " << tau.decayMode() << ")"
            << ", nChHadrs = " << chargedHadrons.size() << ", nLostTracks = " << lostTracks.size();
      }
      if (!lostTracks.empty())
        tau.setsignalTracks(lostTracks);
    }

    void PFRecoTauLostTrackPlugin::endEvent() {}

  }  // namespace tau
}  // namespace reco

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_EDM_PLUGIN(RecoTauModifierPluginFactory, reco::tau::PFRecoTauLostTrackPlugin, "PFRecoTauLostTrackPlugin");
