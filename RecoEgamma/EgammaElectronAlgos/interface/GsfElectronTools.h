#ifndef RecoEgamma_EgammaElectronAlgos_GsfElectronTools_h
#define RecoEgamma_EgammaElectronAlgos_GsfElectronTools_h

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "RecoEgamma/EgammaTools/interface/LazyResult.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"

namespace egamma {

  struct TrackVariables {
    explicit TrackVariables(reco::TrackBase const& tk) : eta{tk.eta()}, phi{tk.phi()} {}
    const double eta;
    const double phi;
  };

  std::vector<TrackVariables> getTrackVariables(reco::TrackCollection const& tracks);

  inline auto getTrackVariablesLazy(reco::TrackCollection const& tracks) {
    return LazyResult(&getTrackVariables, tracks);
  }

  // From Puneeth Kalavase : returns the CTF track that has the highest fraction
  // of shared hits in Pixels and the inner strip tracker with the electron Track
  std::pair<reco::TrackRef, float> getClosestCtfToGsf(reco::GsfTrackRef const&,
                                                      edm::Handle<reco::TrackCollection> const& ctfTracksH,
                                                      std::vector<TrackVariables> const& ctfTrackVariables);

}  // namespace egamma

#endif
