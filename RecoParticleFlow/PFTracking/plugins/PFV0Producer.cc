#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFV0.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoParticleFlow/PFTracking/interface/PFTrackTransformer.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

class PFV0Producer : public edm::stream::EDProducer<> {
public:
  ///Constructor
  explicit PFV0Producer(const edm::ParameterSet&);

  ///Destructor
  ~PFV0Producer() override;

private:
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override;

  ///Produce the PFRecTrack collection
  void produce(edm::Event&, const edm::EventSetup&) override;

  ///PFTrackTransformer
  PFTrackTransformer* pfTransformer_;
  std::vector<edm::EDGetTokenT<reco::VertexCompositeCandidateCollection> > V0list_;

  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldToken_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFV0Producer);

using namespace std;
using namespace edm;
using namespace reco;
PFV0Producer::PFV0Producer(const ParameterSet& iConfig)
    : pfTransformer_(nullptr), magneticFieldToken_(esConsumes<edm::Transition::BeginRun>()) {
  produces<reco::PFV0Collection>();
  produces<reco::PFRecTrackCollection>();

  std::vector<edm::InputTag> tags = iConfig.getParameter<vector<InputTag> >("V0List");

  for (unsigned int i = 0; i < tags.size(); ++i)
    V0list_.push_back(consumes<reco::VertexCompositeCandidateCollection>(tags[i]));
}

PFV0Producer::~PFV0Producer() { delete pfTransformer_; }

void PFV0Producer::produce(Event& iEvent, const EventSetup& iSetup) {
  LogDebug("PFV0Producer") << "START event: " << iEvent.id().event() << " in run " << iEvent.id().run();
  //create the empty collections
  auto pfV0Coll = std::make_unique<PFV0Collection>();

  auto pfV0RecTrackColl = std::make_unique<reco::PFRecTrackCollection>();

  reco::PFRecTrackRefProd pfTrackRefProd = iEvent.getRefBeforePut<reco::PFRecTrackCollection>();
  int idx = 0;

  for (unsigned int il = 0; il < V0list_.size(); il++) {
    Handle<VertexCompositeCandidateCollection> V0coll;
    iEvent.getByToken(V0list_[il], V0coll);
    LogDebug("PFV0Producer") << "V0list_[" << il << "] contains " << V0coll->size() << " V0 candidates ";
    for (unsigned int iv = 0; iv < V0coll->size(); iv++) {
      VertexCompositeCandidateRef V0(V0coll, iv);
      vector<TrackRef> Tracks;
      vector<PFRecTrackRef> PFTracks;
      for (unsigned int ndx = 0; ndx < V0->numberOfDaughters(); ndx++) {
        Tracks.push_back((dynamic_cast<const RecoChargedCandidate*>(V0->daughter(ndx)))->track());
        TrackRef trackRef = (dynamic_cast<const RecoChargedCandidate*>(V0->daughter(ndx)))->track();

        reco::PFRecTrack pfRecTrack(trackRef->charge(), reco::PFRecTrack::KF, trackRef.key(), trackRef);

        Trajectory FakeTraj;
        bool valid = pfTransformer_->addPoints(pfRecTrack, *trackRef, FakeTraj);
        if (valid) {
          PFTracks.push_back(reco::PFRecTrackRef(pfTrackRefProd, idx++));
          pfV0RecTrackColl->push_back(pfRecTrack);
        }
      }
      if ((PFTracks.size() == 2) && (Tracks.size() == 2)) {
        pfV0Coll->push_back(PFV0(V0, PFTracks, Tracks));
      }
    }
  }

  iEvent.put(std::move(pfV0Coll));
  iEvent.put(std::move(pfV0RecTrackColl));
}

// ------------ method called once each job just before starting event loop  ------------
void PFV0Producer::beginRun(const edm::Run& run, const EventSetup& iSetup) {
  auto const& magneticField = &iSetup.getData(magneticFieldToken_);
  pfTransformer_ = new PFTrackTransformer(math::XYZVector(magneticField->inTesla(GlobalPoint(0, 0, 0))));
  pfTransformer_->OnlyProp();
}

// ------------ method called once each job just after ending the event loop  ------------
void PFV0Producer::endRun(const edm::Run& run, const EventSetup& iSetup) {
  delete pfTransformer_;
  pfTransformer_ = nullptr;
}
