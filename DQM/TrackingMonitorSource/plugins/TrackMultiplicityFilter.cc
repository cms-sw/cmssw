// system include files
#include <memory>
#include <algorithm>

// user include files
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// class declaration
//

class TrackMultiplicityFilter : public edm::global::EDFilter<> {
public:
  explicit TrackMultiplicityFilter(const edm::ParameterSet&);
  ~TrackMultiplicityFilter() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  const edm::InputTag tracksTag_;
  const edm::EDGetTokenT<reco::TrackCollection> tracksToken_;
  const StringCutObjectSelector<reco::Track> selector_;
  const unsigned int nmin_;
};

using namespace std;
using namespace edm;

void TrackMultiplicityFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<edm::InputTag>("trackInputTag", edm::InputTag("generalTracks"));
  desc.addUntracked<std::string>("cut", std::string(""));
  desc.addUntracked<uint32_t>("nmin", 0.);
  descriptions.addWithDefaultLabel(desc);
}

TrackMultiplicityFilter::TrackMultiplicityFilter(const edm::ParameterSet& ps)
    : tracksTag_(ps.getUntrackedParameter<edm::InputTag>("trackInputTag", edm::InputTag("generalTracks"))),
      tracksToken_(consumes<reco::TrackCollection>(tracksTag_)),
      selector_(ps.getUntrackedParameter<std::string>("cut", "")),
      nmin_(ps.getUntrackedParameter<uint32_t>("nmin", 0.)) {}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool TrackMultiplicityFilter::filter(edm::StreamID iStream, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  bool pass = false;
  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByToken(tracksToken_, tracks);
  double count = std::count_if(tracks->begin(), tracks->end(), selector_);
  pass = (count >= nmin_);

  edm::LogInfo("TrackMultiplicityFilter") << "pass : " << pass;

  return pass;
}
//define this as a plug-in
DEFINE_FWK_MODULE(TrackMultiplicityFilter);
