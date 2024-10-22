#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include <iostream>
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class CSCTFFilter : public edm::global::EDFilter<> {
public:
  explicit CSCTFFilter(const edm::ParameterSet&);

private:
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  std::vector<unsigned> modes;
  edm::EDGetTokenT<L1CSCTrackCollection> token_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CSCTFFilter::CSCTFFilter(const edm::ParameterSet& iConfig) {
  //now do what ever initialization is needed
  modes = iConfig.getUntrackedParameter<std::vector<unsigned> >("modes");
  token_ = consumes(iConfig.getUntrackedParameter<edm::InputTag>("inputTag"));
}

// member functions
//

// ------------ method called to for each event  ------------
bool CSCTFFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;
  using namespace std;

  edm::Handle<L1CSCTrackCollection> trackFinderTracks = iEvent.getHandle(token_);

  for (auto BaseTFTrk = trackFinderTracks->begin(); BaseTFTrk != trackFinderTracks->end(); BaseTFTrk++) {
    for (auto mode = modes.begin(); mode != modes.end(); mode++) {
      if (BaseTFTrk->first.mode() == (*mode)) {
        //cout << "mode: "<< *mode << endl;
        return true;
      }
    }
  }
  return false;
}

//define this as a plug-in
DEFINE_FWK_MODULE(CSCTFFilter);
