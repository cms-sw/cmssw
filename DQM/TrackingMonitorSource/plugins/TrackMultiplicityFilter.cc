#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TLorentzVector.h"
//#include "RecoEgamma/ElectronIdentification/interface/CutBasedElectronID.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"
#include "DQM/TrackingMonitorSource/interface/TrackMultiplicityFilter.h"

using namespace std;
using namespace edm;

TrackMultiplicityFilter::TrackMultiplicityFilter(const edm::ParameterSet& ps)
    : parameters_(ps),
      tracksTag_(parameters_.getUntrackedParameter<edm::InputTag>("trackInputTag", edm::InputTag("generalTracks"))),
      tracksToken_(consumes<reco::TrackCollection>(tracksTag_)),
      selector_(parameters_.getUntrackedParameter<std::string>("cut", "")),
      nmin_(parameters_.getUntrackedParameter<uint32_t>("nmin", 0.)) {
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
  //  unsigned int count = 0;
  //  for ( auto const &track : *tracks )
  //    if ( selector_(track) ) count++;
  //unsigned int count = std::count_if(tracks->begin(), tracks->end(), selector_);
  double count = std::count_if(tracks->begin(), tracks->end(), selector_);
  std::cout << "count : " << count << std::endl;
  std::cout << "nmin_ : " << nmin_ << std::endl;
  std::cout << "tracks size : " << tracks->size() << std::endl;

  pass = (count >= nmin_);

  std::cout << "pass : " << pass << std::endl;

  return pass;
}
//define this as a plug-in
DEFINE_FWK_MODULE(TrackMultiplicityFilter);
