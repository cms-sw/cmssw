#include <memory>
#include "RecoParticleFlow/PFTracking/interface/LightPFTrackProducer.h"
#include "RecoParticleFlow/PFTracking/interface/PFTrackTransformer.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

using namespace std;
using namespace edm;
LightPFTrackProducer::LightPFTrackProducer(const ParameterSet& iConfig):
  pfTransformer_(0)
{
  produces<reco::PFRecTrackCollection>();


  
  std::vector<InputTag>  tags = 
    iConfig.getParameter< vector < InputTag > >("TkColList");

  for (unsigned int i=0;i<tags.size();++i)
    tracksContainers_.push_back(consumes<reco::TrackCollection>(tags[i]));

  useQuality_   = iConfig.getParameter<bool>("UseQuality");
  trackQuality_=reco::TrackBase::qualityByName(iConfig.getParameter<std::string>("TrackQuality"));

}

LightPFTrackProducer::~LightPFTrackProducer()
{
  delete pfTransformer_;
}

void
LightPFTrackProducer::produce(Event& iEvent, const EventSetup& iSetup)
{

  //create the empty collections 
  auto_ptr< reco::PFRecTrackCollection > 
    PfTrColl (new reco::PFRecTrackCollection);
  
  for (unsigned int istr=0; istr<tracksContainers_.size();istr++){
    
    //Track collection
    Handle<reco::TrackCollection> tkRefCollection;
    iEvent.getByToken(tracksContainers_[istr], tkRefCollection);
    reco::TrackCollection  Tk=*(tkRefCollection.product());
    for(unsigned int i=0;i<Tk.size();i++){
      if (useQuality_ &&
	  (!(Tk[i].quality(trackQuality_)))) continue;
     reco::TrackRef trackRef(tkRefCollection, i);
      reco::PFRecTrack pftrack( trackRef->charge(), 
       				reco::PFRecTrack::KF, 
       				i, trackRef );
      Trajectory FakeTraj;
      bool mymsgwarning = false;
      bool valid = pfTransformer_->addPoints( pftrack, *trackRef, FakeTraj, mymsgwarning);
      if(valid)
	PfTrColl->push_back(pftrack);		

    }
  }
  iEvent.put(PfTrColl);
}

// ------------ method called once each job just before starting event loop  ------------
void 
LightPFTrackProducer::beginRun(const edm::Run& run,
			       const EventSetup& iSetup)
{
  ESHandle<MagneticField> magneticField;
  iSetup.get<IdealMagneticFieldRecord>().get(magneticField);
  pfTransformer_= new PFTrackTransformer(math::XYZVector(magneticField->inTesla(GlobalPoint(0,0,0))));
  pfTransformer_->OnlyProp();
}

// ------------ method called once each job just after ending the event loop  ------------
void 
LightPFTrackProducer::endRun(const edm::Run& run,
			     const EventSetup& iSetup) {
  delete pfTransformer_;
  pfTransformer_=nullptr;
}
