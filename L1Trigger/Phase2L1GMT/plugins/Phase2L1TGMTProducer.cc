#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"
#include "Node.h"

//
// class declaration
//
using namespace Phase2L1GMT;
using namespace l1t;

class Phase2L1TGMTProducer : public edm::stream::EDProducer<> {
public:
  explicit Phase2L1TGMTProducer(const edm::ParameterSet&);
  ~Phase2L1TGMTProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;
  std::unique_ptr<Node> node_;
  edm::EDGetTokenT<l1t::TrackerMuon::L1TTTrackCollection> srcTracks_;
  edm::EDGetTokenT<std::vector<l1t::MuonStub> > srcStubs_;
  edm::EDGetTokenT<BXVector<RegionalMuonCand> > bmtfTracks_;
  edm::EDGetTokenT<BXVector<RegionalMuonCand> > emtfTracks_;
  edm::EDGetTokenT<BXVector<RegionalMuonCand> > omtfTracks_;
  int minTrackStubs_;
  int bxMin_;
  int bxMax_;
};

Phase2L1TGMTProducer::Phase2L1TGMTProducer(const edm::ParameterSet& iConfig)
    : node_(new Node(iConfig)),
      srcTracks_(consumes<l1t::TrackerMuon::L1TTTrackCollection>(iConfig.getParameter<edm::InputTag>("srcTracks"))),
      srcStubs_(consumes<std::vector<l1t::MuonStub> >(iConfig.getParameter<edm::InputTag>("srcStubs"))),
      bmtfTracks_(consumes<BXVector<RegionalMuonCand> >(iConfig.getParameter<edm::InputTag>("srcBMTF"))),
      emtfTracks_(consumes<BXVector<RegionalMuonCand> >(iConfig.getParameter<edm::InputTag>("srcEMTF"))),
      omtfTracks_(consumes<BXVector<RegionalMuonCand> >(iConfig.getParameter<edm::InputTag>("srcOMTF"))),
      minTrackStubs_(iConfig.getParameter<int>("minTrackStubs")),
      bxMin_(iConfig.getParameter<int>("muonBXMin")),
      bxMax_(iConfig.getParameter<int>("muonBXMax"))

{
  produces<std::vector<l1t::TrackerMuon> >();
}

Phase2L1TGMTProducer::~Phase2L1TGMTProducer() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void Phase2L1TGMTProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
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

  Handle<std::vector<l1t::MuonStub> > stubHandle;
  iEvent.getByToken(srcStubs_, stubHandle);
  l1t::MuonStubRefVector stubs;
  for (uint i = 0; i < stubHandle->size(); ++i) {
    l1t::MuonStubRef stub(stubHandle, i);
    if (stub->bxNum() >= bxMin_ && stub->bxNum() <= bxMax_)
      stubs.push_back(stub);
  }

  ObjectRefBxCollection<RegionalMuonCand> muonTracks;
  //   BXVector<RegionalMuonCand> muonTracks;

  Handle<BXVector<RegionalMuonCand> > bmtfHandle;
  iEvent.getByToken(bmtfTracks_, bmtfHandle);
  Handle<BXVector<RegionalMuonCand> > emtfHandle;
  iEvent.getByToken(emtfTracks_, emtfHandle);
  Handle<BXVector<RegionalMuonCand> > omtfHandle;
  iEvent.getByToken(omtfTracks_, omtfHandle);

  for (int bx = bxMin_; bx <= bxMax_; ++bx) {
    if (bx >= bmtfHandle->getFirstBX() && bx <= bmtfHandle->getLastBX())
      for (size_t i = 0; i < bmtfHandle->size(bx); ++i) {
        RegionalMuonCandRef muon(bmtfHandle, i);
        muonTracks.push_back(bx, muon);
      }
    if (bx >= omtfHandle->getFirstBX() && bx <= omtfHandle->getLastBX())
      for (size_t i = 0; i < omtfHandle->size(bx); ++i) {
        RegionalMuonCandRef muon(omtfHandle, i);
        muonTracks.push_back(bx, muon);
      }
    if (bx >= emtfHandle->getFirstBX() && bx <= emtfHandle->getLastBX())
      for (size_t i = 0; i < emtfHandle->size(bx); ++i) {
        RegionalMuonCandRef muon(emtfHandle, i);
        muonTracks.push_back(bx, muon);
      }
  }

  std::vector<l1t::TrackerMuon> out = node_->processEvent(tracks, muonTracks, stubs);
  std::unique_ptr<std::vector<l1t::TrackerMuon> > out1 = std::make_unique<std::vector<l1t::TrackerMuon> >(out);
  iEvent.put(std::move(out1));
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void Phase2L1TGMTProducer::beginStream(edm::StreamID) {}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void Phase2L1TGMTProducer::endStream() {}

void Phase2L1TGMTProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(Phase2L1TGMTProducer);
