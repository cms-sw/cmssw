#include <memory>
#include "RecoParticleFlow/PFTracking/interface/LightPFTrackProducer.h"
#include "RecoParticleFlow/PFTracking/interface/PFTrackTransformer.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
using namespace std;
using namespace edm;
LightPFTrackProducer::LightPFTrackProducer(const ParameterSet& iConfig):
  pfTransformer_(0)
{
  produces<reco::PFRecTrackCollection>();

  tracksContainers_ = 
    iConfig.getParameter< vector < InputTag > >("TkColList");
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
  
  for (uint istr=0; istr<tracksContainers_.size();istr++){
    
    //Track collection
    Handle<reco::TrackCollection> tkRefCollection;
    iEvent.getByLabel(tracksContainers_[istr], tkRefCollection);
    reco::TrackCollection  Tk=*(tkRefCollection.product());
    for(uint i=0;i<Tk.size();i++){
      reco::TrackRef trackRef(tkRefCollection, i);
      reco::PFRecTrack pftrack( trackRef->charge(), 
       				reco::PFRecTrack::KF, 
       				i, trackRef );
      Trajectory FakeTraj;
      bool valid = pfTransformer_->addPoints( pftrack, *trackRef, FakeTraj);
      if(valid)
	PfTrColl->push_back(pftrack);		

    }
  }
  iEvent.put(PfTrColl);
}

// ------------ method called once each job just before starting event loop  ------------
void 
LightPFTrackProducer::beginJob(const EventSetup& iSetup)
{
  pfTransformer_= new PFTrackTransformer();
  pfTransformer_->OnlyProp();
}

// ------------ method called once each job just after ending the event loop  ------------
void 
LightPFTrackProducer::endJob() {
}
