/* \class PtMaxTrackCountFilter
 *
 * Filters events if at least N tracks below
 * a pt cut are present.
 *
 * \author: Marco Musich
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "CommonTools/UtilAlgos/interface/ObjectCountFilter.h"
#include "CommonTools/UtilAlgos/interface/PtMaxSelector.h"

typedef ObjectCountFilter<reco::TrackCollection, PtMaxSelector>::type PtMaxTrackCountFilter;

template <>
void PtMaxTrackCountFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("tracks"));
  desc.add<unsigned int>("minNumber", 1);
  desc.add<double>("ptMax", 999.);
  desc.add<std::string>("cut", "");
  descriptions.add("ptMaxTrackCountFilter", desc);
}

DEFINE_FWK_MODULE(PtMaxTrackCountFilter);
