#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoParticleFlow/PFTracking/interface/PFTrackTransformer.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

class LightPFTrackProducer : public edm::stream::EDProducer<> {
public:
  ///Constructor
  explicit LightPFTrackProducer(const edm::ParameterSet&);

  ///Destructor
  ~LightPFTrackProducer() override;

private:
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override;

  ///Produce the PFRecTrack collection
  void produce(edm::Event&, const edm::EventSetup&) override;

  ///PFTrackTransformer
  PFTrackTransformer* pfTransformer_;
  std::vector<edm::EDGetTokenT<reco::TrackCollection> > tracksContainers_;

  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldToken_;
  ///TRACK QUALITY
  bool useQuality_;
  reco::TrackBase::TrackQuality trackQuality_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LightPFTrackProducer);

using namespace std;
using namespace edm;
LightPFTrackProducer::LightPFTrackProducer(const ParameterSet& iConfig)
    : pfTransformer_(nullptr), magneticFieldToken_(esConsumes<edm::Transition::BeginRun>()) {
  produces<reco::PFRecTrackCollection>();

  std::vector<InputTag> tags = iConfig.getParameter<vector<InputTag> >("TkColList");

  for (unsigned int i = 0; i < tags.size(); ++i)
    tracksContainers_.push_back(consumes<reco::TrackCollection>(tags[i]));

  useQuality_ = iConfig.getParameter<bool>("UseQuality");
  trackQuality_ = reco::TrackBase::qualityByName(iConfig.getParameter<std::string>("TrackQuality"));
}

LightPFTrackProducer::~LightPFTrackProducer() { delete pfTransformer_; }

void LightPFTrackProducer::produce(Event& iEvent, const EventSetup& iSetup) {
  //create the empty collections
  auto PfTrColl = std::make_unique<reco::PFRecTrackCollection>();

  for (unsigned int istr = 0; istr < tracksContainers_.size(); istr++) {
    //Track collection
    Handle<reco::TrackCollection> tkRefCollection;
    iEvent.getByToken(tracksContainers_[istr], tkRefCollection);
    reco::TrackCollection Tk = *(tkRefCollection.product());
    for (unsigned int i = 0; i < Tk.size(); i++) {
      if (useQuality_ && (!(Tk[i].quality(trackQuality_))))
        continue;
      reco::TrackRef trackRef(tkRefCollection, i);
      reco::PFRecTrack pftrack(trackRef->charge(), reco::PFRecTrack::KF, i, trackRef);
      Trajectory FakeTraj;
      bool mymsgwarning = false;
      bool valid = pfTransformer_->addPoints(pftrack, *trackRef, FakeTraj, mymsgwarning);
      if (valid)
        PfTrColl->push_back(pftrack);
    }
  }
  iEvent.put(std::move(PfTrColl));
}

// ------------ method called once each job just before starting event loop  ------------
void LightPFTrackProducer::beginRun(const edm::Run& run, const EventSetup& iSetup) {
  auto const& magneticField = &iSetup.getData(magneticFieldToken_);
  pfTransformer_ = new PFTrackTransformer(math::XYZVector(magneticField->inTesla(GlobalPoint(0, 0, 0))));
  pfTransformer_->OnlyProp();
}

// ------------ method called once each job just after ending the event loop  ------------
void LightPFTrackProducer::endRun(const edm::Run& run, const EventSetup& iSetup) {
  delete pfTransformer_;
  pfTransformer_ = nullptr;
}
