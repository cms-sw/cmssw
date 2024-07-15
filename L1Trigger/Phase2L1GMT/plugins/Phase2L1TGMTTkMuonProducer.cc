#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"
#include "L1Trigger/Phase2L1GMT/interface/TPS.h"

//
// class declaration
//
using namespace Phase2L1GMT;
using namespace l1t;

class Phase2L1TGMTTkMuonProducer : public edm::stream::EDProducer<> {
public:
  explicit Phase2L1TGMTTkMuonProducer(const edm::ParameterSet&);
  ~Phase2L1TGMTTkMuonProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;
  std::unique_ptr<TPS> tps_;
  edm::EDGetTokenT<l1t::TrackerMuon::L1TTTrackCollection> srcTracks_;
  edm::EDGetTokenT<std::vector<l1t::MuonStub> > srcStubs_;
  int minTrackStubs_;
  int bxMin_;
  int bxMax_;
};

Phase2L1TGMTTkMuonProducer::Phase2L1TGMTTkMuonProducer(const edm::ParameterSet& iConfig)
    : tps_(new TPS(iConfig)),
      srcTracks_(consumes<l1t::TrackerMuon::L1TTTrackCollection>(iConfig.getParameter<edm::InputTag>("srcTracks"))),
      srcStubs_(consumes<std::vector<l1t::MuonStub> >(iConfig.getParameter<edm::InputTag>("srcStubs"))),
      minTrackStubs_(iConfig.getParameter<int>("minTrackStubs")),
      bxMin_(iConfig.getParameter<int>("muonBXMin")),
      bxMax_(iConfig.getParameter<int>("muonBXMax"))

{
  produces<std::vector<l1t::TrackerMuon> >();
}

Phase2L1TGMTTkMuonProducer::~Phase2L1TGMTTkMuonProducer() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void Phase2L1TGMTTkMuonProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  Handle<l1t::TrackerMuon::L1TTTrackCollection> trackHandle;
  iEvent.getByToken(srcTracks_, trackHandle);
  std::vector<edm::Ptr<l1t::TrackerMuon::L1TTTrackType> > tracks;
  for (uint i = 0; i < trackHandle->size(); ++i) {
    edm::Ptr<l1t::TrackerMuon::L1TTTrackType> track(trackHandle, i);
    if (track->momentum().transverse() < 2.0)
      continue;
    if (track->getStubRefs().size() >= (unsigned int)(minTrackStubs_))
      tracks.push_back(track);
  }

  l1t::MuonStubRefVector muonStubs;
  Handle<std::vector<l1t::MuonStub> > stubHandle;
  iEvent.getByToken(srcStubs_, stubHandle);
  for (size_t i = 0; i < stubHandle->size(); ++i) {
    MuonStubRef stub(stubHandle, i);
    muonStubs.push_back(stub);
  }

  std::vector<l1t::TrackerMuon> out = tps_->processEvent(tracks, muonStubs);
  std::unique_ptr<std::vector<l1t::TrackerMuon> > out1 = std::make_unique<std::vector<l1t::TrackerMuon> >(out);
  iEvent.put(std::move(out1));
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void Phase2L1TGMTTkMuonProducer::beginStream(edm::StreamID) {}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void Phase2L1TGMTTkMuonProducer::endStream() {}

void Phase2L1TGMTTkMuonProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(Phase2L1TGMTTkMuonProducer);
