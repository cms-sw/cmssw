#ifndef RecoEgamma_EgammaElectronAlgos_GsfElectronTools_h
#define RecoEgamma_EgammaElectronAlgos_GsfElectronTools_h

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

namespace egamma {

  // From Puneeth Kalavase : returns the CTF track that has the highest fraction
  // of shared hits in Pixels and the inner strip tracker with the electron Track
  std::pair<reco::TrackRef, float> getClosestCtfToGsf(reco::GsfTrackRef const&,
                                                      edm::Handle<reco::TrackCollection> const& ctfTracksH);

}  // namespace egamma

#endif
