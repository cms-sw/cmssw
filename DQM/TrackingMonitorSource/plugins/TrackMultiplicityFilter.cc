#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"
#include "DQM/TrackingMonitorSource/interface/TrackMultiplicityFilter.h"

using namespace std;
using namespace edm;

TrackMultiplicityFilter::TrackMultiplicityFilter(const edm::ParameterSet& ps)
    : tracksTag_(ps.getUntrackedParameter<edm::InputTag>("trackInputTag", edm::InputTag("generalTracks"))),
      tracksToken_(consumes<reco::TrackCollection>(tracksTag_)),
      selector_(ps.getUntrackedParameter<std::string>("cut", "")),
      nmin_(ps.getUntrackedParameter<uint32_t>("nmin", 0.)) {
  //now do what ever initialization is needed
}

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
