#include "DataFormats/ParticleFlowReco/interface/PFDisplacedTrackerVertex.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoParticleFlow/PFTracking/interface/PFTrackTransformer.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

class PFDisplacedTrackerVertexProducer : public edm::stream::EDProducer<> {
public:
  ///Constructor
  explicit PFDisplacedTrackerVertexProducer(const edm::ParameterSet&);

  ///Destructor
  ~PFDisplacedTrackerVertexProducer() override;

private:
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override;

  ///Produce the PFRecTrack collection
  void produce(edm::Event&, const edm::EventSetup&) override;

  ///PFTrackTransformer
  PFTrackTransformer* pfTransformer_;
  edm::EDGetTokenT<reco::PFDisplacedVertexCollection> pfDisplacedVertexContainer_;
  edm::EDGetTokenT<reco::TrackCollection> pfTrackContainer_;

  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldToken_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFDisplacedTrackerVertexProducer);

using namespace std;
using namespace edm;
PFDisplacedTrackerVertexProducer::PFDisplacedTrackerVertexProducer(const ParameterSet& iConfig)
    : pfTransformer_(nullptr), magneticFieldToken_(esConsumes<edm::Transition::BeginRun>()) {
  produces<reco::PFRecTrackCollection>();
  produces<reco::PFDisplacedTrackerVertexCollection>();

  pfDisplacedVertexContainer_ =
      consumes<reco::PFDisplacedVertexCollection>(iConfig.getParameter<InputTag>("displacedTrackerVertexColl"));

  pfTrackContainer_ = consumes<reco::TrackCollection>(iConfig.getParameter<InputTag>("trackColl"));
}

PFDisplacedTrackerVertexProducer::~PFDisplacedTrackerVertexProducer() { delete pfTransformer_; }

void PFDisplacedTrackerVertexProducer::produce(Event& iEvent, const EventSetup& iSetup) {
  //create the empty collections
  auto pfDisplacedTrackerVertexColl = std::make_unique<reco::PFDisplacedTrackerVertexCollection>();
  auto pfRecTrackColl = std::make_unique<reco::PFRecTrackCollection>();

  reco::PFRecTrackRefProd pfTrackRefProd = iEvent.getRefBeforePut<reco::PFRecTrackCollection>();

  Handle<reco::PFDisplacedVertexCollection> nuclCollH;
  iEvent.getByToken(pfDisplacedVertexContainer_, nuclCollH);
  const reco::PFDisplacedVertexCollection& nuclColl = *(nuclCollH.product());

  Handle<reco::TrackCollection> trackColl;
  iEvent.getByToken(pfTrackContainer_, trackColl);

  int idx = 0;

  //  cout << "Size of Displaced Vertices "
  //     <<  nuclColl.size() << endl;

  // loop on all NuclearInteraction
  for (unsigned int icoll = 0; icoll < nuclColl.size(); icoll++) {
    reco::PFRecTrackRefVector pfRecTkcoll;

    std::vector<reco::Track> refittedTracks = nuclColl[icoll].refittedTracks();

    // convert the secondary tracks
    for (unsigned it = 0; it < refittedTracks.size(); it++) {
      reco::TrackBaseRef trackBaseRef = nuclColl[icoll].originalTrack(refittedTracks[it]);

      //      cout << "track base pt = " << trackBaseRef->pt() << endl;

      reco::TrackRef trackRef(trackColl, trackBaseRef.key());

      //      cout << "track pt = " << trackRef->pt() << endl;

      reco::PFRecTrack pfRecTrack(trackBaseRef->charge(), reco::PFRecTrack::KF, trackBaseRef.key(), trackRef);

      // cout << pfRecTrack << endl;

      Trajectory FakeTraj;
      bool valid = pfTransformer_->addPoints(pfRecTrack, *trackBaseRef, FakeTraj);
      if (valid) {
        pfRecTkcoll.push_back(reco::PFRecTrackRef(pfTrackRefProd, idx++));
        pfRecTrackColl->push_back(pfRecTrack);
        //	cout << "after "<< pfRecTrack << endl;
      }
    }
    reco::PFDisplacedVertexRef niRef(nuclCollH, icoll);
    pfDisplacedTrackerVertexColl->push_back(reco::PFDisplacedTrackerVertex(niRef, pfRecTkcoll));
  }

  iEvent.put(std::move(pfRecTrackColl));
  iEvent.put(std::move(pfDisplacedTrackerVertexColl));
}

// ------------ method called once each job just before starting event loop  ------------
void PFDisplacedTrackerVertexProducer::beginRun(const edm::Run& run, const EventSetup& iSetup) {
  auto const& magneticField = &iSetup.getData(magneticFieldToken_);
  pfTransformer_ = new PFTrackTransformer(math::XYZVector(magneticField->inTesla(GlobalPoint(0, 0, 0))));
  pfTransformer_->OnlyProp();
}

// ------------ method called once each job just after ending the event loop  ------------
void PFDisplacedTrackerVertexProducer::endRun(const edm::Run& run, const EventSetup& iSetup) {
  delete pfTransformer_;
  pfTransformer_ = nullptr;
}
