/* \class PtMinTrackCountFilter
 *
 * Filters events if at least N tracks above 
 * a pt cut are present.
 *
 * \author: Antonio Vagnerini, DESY
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "CommonTools/UtilAlgos/interface/ObjectCountFilter.h"
#include "CommonTools/UtilAlgos/interface/PtMinSelector.h"

typedef ObjectCountFilter<reco::TrackCollection, PtMinSelector>::type PtMinTrackCountFilter;

template <>
void PtMinTrackCountFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("tracks"));
  desc.add<double>("ptMin", 0.);
  desc.add<std::string>("cut", "");
  descriptions.add("ptMinTrackCountFilter", desc);
}

DEFINE_FWK_MODULE(PtMinTrackCountFilter);
