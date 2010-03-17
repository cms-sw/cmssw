#include <memory>
#include "RecoParticleFlow/PFTracking/plugins/PFConversionProducer.h"
#include "RecoParticleFlow/PFTracking/interface/PFTrackTransformer.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversionFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversion.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
using namespace std;
using namespace edm;
PFConversionProducer::PFConversionProducer(const ParameterSet& iConfig):
  pfTransformer_(0)
{
  produces<reco::PFRecTrackCollection>();
  produces<reco::PFConversionCollection>();

  pfConversionContainer_ = 
    iConfig.getParameter< InputTag >("conversionCollection");

}

PFConversionProducer::~PFConversionProducer()
{
  delete pfTransformer_;
}

void
PFConversionProducer::produce(Event& iEvent, const EventSetup& iSetup)
{

  //create the empty collections 
  auto_ptr< reco::PFConversionCollection > 
    pfConversionColl (new reco::PFConversionCollection);
  auto_ptr< reco::PFRecTrackCollection > 
    pfRecTrackColl (new reco::PFRecTrackCollection);
  
  reco::PFRecTrackRefProd pfTrackRefProd = iEvent.getRefBeforePut<reco::PFRecTrackCollection>();


    
  Handle<reco::ConversionCollection> convCollH;
  iEvent.getByLabel(pfConversionContainer_, convCollH);
  const reco::ConversionCollection& convColl = *(convCollH.product());

  Handle<reco::TrackCollection> trackColl;
  iEvent.getByLabel(pfTrackContainer_, trackColl);

  int idx = 0;

  cout << "Size of Displaced Vertices " 
       <<  convColl.size() << endl;

  // loop on all NuclearInteraction 
  for( uint icoll=0; icoll < convColl.size(); icoll++) {

    std::vector<reco::PFRecTrackRef> pfRecTkcoll;

    std::vector<reco::TrackRef> tracksRefColl = convColl[icoll].tracks();

    // convert the secondary tracks
    for(int it = 0; it < tracksRefColl.size(); it++){

      reco::TrackRef trackRef = tracksRefColl[it];

      reco::PFRecTrack pfRecTrack( trackRef->charge(), 
				   reco::PFRecTrack::KF, 
				   trackRef.key(), 
				   trackRef );

      // cout << pfRecTrack << endl;

      Trajectory FakeTraj;
      bool valid = pfTransformer_->addPoints( pfRecTrack, *trackRef, FakeTraj);
      if(valid) {
	pfRecTkcoll.push_back(reco::PFRecTrackRef( pfTrackRefProd, idx++));	
	pfRecTrackColl->push_back(pfRecTrack);
	//	cout << "after "<< pfRecTrack << endl;
          
      }
    }
    reco::ConversionRef niRef(convCollH, icoll);
    pfConversionColl->push_back( reco::PFConversion( niRef, pfRecTkcoll ));
  }
 
  iEvent.put(pfRecTrackColl);
  iEvent.put(pfConversionColl);
}

// ------------ method called once each job just before starting event loop  ------------
void 
PFConversionProducer::beginRun(edm::Run& run,
					   const EventSetup& iSetup)
{
  ESHandle<MagneticField> magneticField;
  iSetup.get<IdealMagneticFieldRecord>().get(magneticField);
  pfTransformer_= new PFTrackTransformer(math::XYZVector(magneticField->inTesla(GlobalPoint(0,0,0))));
  pfTransformer_->OnlyProp();
}

// ------------ method called once each job just after ending the event loop  ------------
void 
PFConversionProducer::endRun() {
  delete pfTransformer_;
}
