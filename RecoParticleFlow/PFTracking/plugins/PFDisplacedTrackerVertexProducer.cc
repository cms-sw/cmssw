#include <memory>
#include "RecoParticleFlow/PFTracking/plugins/PFDisplacedTrackerVertexProducer.h"
#include "RecoParticleFlow/PFTracking/interface/PFTrackTransformer.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
using namespace std;
using namespace edm;
PFDisplacedTrackerVertexProducer::PFDisplacedTrackerVertexProducer(const ParameterSet& iConfig):
  pfTransformer_(0)
{
  produces<reco::PFRecTrackCollection>();
  produces<reco::PFDisplacedTrackerVertexCollection>();

  pfDisplacedVertexContainer_ = consumes<reco::PFDisplacedVertexCollection>(									    iConfig.getParameter< InputTag >("displacedTrackerVertexColl"));


  pfTrackContainer_ =consumes<reco::TrackCollection>(
    iConfig.getParameter< InputTag >("trackColl"));

}

PFDisplacedTrackerVertexProducer::~PFDisplacedTrackerVertexProducer()
{
  delete pfTransformer_;
}

void
PFDisplacedTrackerVertexProducer::produce(Event& iEvent, const EventSetup& iSetup)
{

  //create the empty collections 
  auto_ptr< reco::PFDisplacedTrackerVertexCollection > 
    pfDisplacedTrackerVertexColl (new reco::PFDisplacedTrackerVertexCollection);
  auto_ptr< reco::PFRecTrackCollection > 
    pfRecTrackColl (new reco::PFRecTrackCollection);
  
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
  for( unsigned int icoll=0; icoll < nuclColl.size(); icoll++) {

    reco::PFRecTrackRefVector pfRecTkcoll;

    std::vector<reco::Track> refittedTracks = nuclColl[icoll].refittedTracks();

    // convert the secondary tracks
    for(unsigned it = 0; it < refittedTracks.size(); it++){

      reco::TrackBaseRef trackBaseRef = nuclColl[icoll].originalTrack(refittedTracks[it]);

      //      cout << "track base pt = " << trackBaseRef->pt() << endl;

      reco::TrackRef trackRef(trackColl, trackBaseRef.key());

      //      cout << "track pt = " << trackRef->pt() << endl;


      reco::PFRecTrack pfRecTrack( trackBaseRef->charge(), 
				   reco::PFRecTrack::KF, 
				   trackBaseRef.key(), 
				   trackRef );

      // cout << pfRecTrack << endl;

      Trajectory FakeTraj;
      bool valid = pfTransformer_->addPoints( pfRecTrack, *trackBaseRef, FakeTraj);
      if(valid) {
	pfRecTkcoll.push_back(reco::PFRecTrackRef( pfTrackRefProd, idx++));	
	pfRecTrackColl->push_back(pfRecTrack);
	//	cout << "after "<< pfRecTrack << endl;
          
      }
    }
    reco::PFDisplacedVertexRef niRef(nuclCollH, icoll);
    pfDisplacedTrackerVertexColl->push_back( reco::PFDisplacedTrackerVertex( niRef, pfRecTkcoll ));
  }
 
  iEvent.put(pfRecTrackColl);
  iEvent.put(pfDisplacedTrackerVertexColl);
}

// ------------ method called once each job just before starting event loop  ------------
void 
PFDisplacedTrackerVertexProducer::beginRun(const edm::Run& run,
					   const EventSetup& iSetup)
{
  ESHandle<MagneticField> magneticField;
  iSetup.get<IdealMagneticFieldRecord>().get(magneticField);
  pfTransformer_= new PFTrackTransformer(math::XYZVector(magneticField->inTesla(GlobalPoint(0,0,0))));
  pfTransformer_->OnlyProp();
}

// ------------ method called once each job just after ending the event loop  ------------
void 
PFDisplacedTrackerVertexProducer::endRun(const edm::Run& run,
					 const EventSetup& iSetup) {
  delete pfTransformer_;
  pfTransformer_=nullptr;
}
