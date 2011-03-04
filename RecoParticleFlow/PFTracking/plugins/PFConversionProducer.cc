#include <memory>
#include "RecoParticleFlow/PFTracking/plugins/PFConversionProducer.h"
#include "RecoParticleFlow/PFTracking/interface/PFTrackTransformer.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversionFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversion.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "RecoEgamma/EgammaTools/interface/ConversionLikelihoodCalculator.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionHitChecker.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/IPTools/interface/IPTools.h"

typedef  std::multimap<unsigned, std::vector<unsigned> > BlockMap;
using namespace std;
using namespace edm;


PFConversionProducer::PFConversionProducer(const ParameterSet& iConfig):
  pfTransformer_(0)
{
  produces<reco::PFRecTrackCollection>();
  produces<reco::PFConversionCollection>();

  pfConversionContainer_ = 
    iConfig.getParameter< InputTag >("conversionCollection");
  vtx_h=iConfig.getParameter<edm::InputTag>("PrimaryVertexLabel");
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
  
  edm::ESHandle<TransientTrackBuilder> builder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", builder);
  TransientTrackBuilder thebuilder = *(builder.product());
  reco::PFRecTrackRefProd pfTrackRefProd = iEvent.getRefBeforePut<reco::PFRecTrackCollection>();
  Handle<reco::ConversionCollection> convCollH;
  iEvent.getByLabel(pfConversionContainer_, convCollH);
  
  const reco::ConversionCollection& convColl = *(convCollH.product());
  
  Handle<reco::TrackCollection> trackColl;
  iEvent.getByLabel(pfTrackContainer_, trackColl);
  Handle<reco::VertexCollection> vertex;
  iEvent.getByLabel(vtx_h, vertex);
  //Find PV for IP calculation, if there is no PV in collection than use dummy 
  reco::Vertex dummy;
  const reco::Vertex* pv=&dummy;    
  if (vertex.isValid()) 
    {
      pv = &*vertex->begin();
    } 
  else 
    { // create a dummy PV
      reco::Vertex::Error e;
      e(0, 0) = 0.0015 * 0.0015;
      e(1, 1) = 0.0015 * 0.0015;
      e(2, 2) = 15. * 15.;
      reco::Vertex::Point p(0, 0, 0);
      dummy = reco::Vertex(p, e, 0, 0, 0);   
    } 
  

  int idx = 0; //index of track in PFRecTrack collection 
  BlockMap trackmap; //Map of Collections and tracks  
  
  //Fill MAP OF ALL CONVERSIONS
  for( unsigned int icoll=0; icoll < convColl.size(); icoll++) //loop over Conversion collections (track pairs)
    {   
      std::vector<unsigned> track_container(0); 
      //take only high purity tracks in merged collection of Ecal-Seeded and general tracks
      if (!( convColl[icoll].quality(reco::Conversion::arbitratedMergedEcalGeneral))||
	  (!convColl[icoll].quality(reco::Conversion::highPurity))) continue;           
      std::vector<edm::RefToBase<reco::Track> > tracksRefColl = convColl[icoll].tracks();   

      for(unsigned it = 0; it < tracksRefColl.size(); it++)
	track_container.push_back(it);	

      //Fill map of collection indices and track indices
      trackmap.insert(make_pair(icoll, track_container)); 
    }
  

  // CLEAN CONVERSION COLLECTION FOR DUPLICATES     
  for( unsigned int icoll=0; icoll < convColl.size(); icoll++) 
    { 
      if (( !convColl[icoll].quality(reco::Conversion::arbitratedMergedEcalGeneral)) || 
	  (!convColl[icoll].quality(reco::Conversion::highPurity))) continue;
      

      std::vector<edm::RefToBase<reco::Track> > tracksRefColl = convColl[icoll].tracks();      

      for(unsigned it = 0; it < tracksRefColl.size(); it++)
	{
	  
	  reco::TrackRef trackRef = (tracksRefColl[it]).castTo<reco::TrackRef>();	
	  for(std::multimap< unsigned, std::vector<unsigned> >::iterator i=trackmap.begin(); i!=trackmap.end(); i++){
	    for(unsigned int j=0; j<(i->second).size(); j++)
	      {
		//comparing Collection to itself
		//if(i->first==icoll)continue; 
		if(i->first==icoll && it==i->second[j])continue; 
		
		std::vector<edm::RefToBase<reco::Track> > tracksRefColl_Check = convColl[i->first].tracks();      
		unsigned int check=i->second[j];
		reco::TrackRef trackcheck = (tracksRefColl_Check[check]).castTo<reco::TrackRef>();

		double like1=-999;
		double like2=-999;
		//number of shared hits
		int shared=0;
		for(trackingRecHit_iterator iHit1=trackRef->recHitsBegin(); iHit1!=trackRef->recHitsEnd(); iHit1++) 
		  {
		    const TrackingRecHit *h_1=iHit1->get();
		    if(h_1->isValid()){		  
		      for(trackingRecHit_iterator iHit2=trackcheck->recHitsBegin(); iHit2!=trackcheck->recHitsEnd(); iHit2++)
			{
			  const TrackingRecHit *h_2=iHit2->get();
			  if(h_2->isValid() && h_1->sharesInput(h_2, TrackingRecHit::some))shared++;//count number of shared hits
			}
		    }
		  }
       
		float frac=0;
		float size1=trackRef->found();
		float size2=trackcheck->found();
		//divide number of shared hits by the total number of hits for the track with less hits
		if(size1>size2)frac=(double)shared/size2;
		else frac=(double)shared/size1;
		if(frac>0.9)
		  {
		    //Calculate Chi^2/ndof Probability for each collection
		    like1=ChiSquaredProbability(convColl[icoll].conversionVertex().chi2(), 
						convColl[icoll].conversionVertex().ndof());
		    like2=ChiSquaredProbability(convColl[i->first].conversionVertex().chi2(), 
						convColl[i->first].conversionVertex().ndof());
		    
		    //delete collection with smaller Probability
		    if(like1>like2)
		      {		    
			std::multimap< unsigned, std::vector<unsigned> >::iterator iter;  
			//find iterator to collection
			iter=trackmap.find(i->first);
			//if found then delete
			if(iter!=trackmap.end())
			  trackmap.erase(iter->first);    	      	      }
		    else 
		      { 
			std::multimap< unsigned, std::vector<unsigned> >::iterator iter;
			iter=trackmap.find(icoll);
			if(iter!=trackmap.end())
			  trackmap.erase(iter->first);
			
		      }
		  } //end if(frac>0.9)				
	      } //end loop over tracks in MAP
	  } //end loop over collections in Map
	} //end loop over tracks in collection
    }//end loop over collections

  //Finally fill empty collections
  for(std::multimap< unsigned, std::vector<unsigned> >::iterator i=trackmap.begin(); i!=trackmap.end(); i++)//looping over Collections in map
    {
      std::vector<reco::PFRecTrackRef> pfRecTkcoll;	
      for(unsigned int j=0; j<(i->second).size(); j++)
	{
	  std::vector<edm::RefToBase<reco::Track> > tracksRefColl = convColl[i->first].tracks();	  
	  // convert the secondary tracks
	 
	  reco::TrackRef trackRef = (tracksRefColl[i->second[j]]).castTo<reco::TrackRef>();
	  
	  reco::PFRecTrack pfRecTrack( trackRef->charge(), 
				       reco::PFRecTrack::KF, 
				       trackRef.key(), 
				       trackRef );             
	  Trajectory FakeTraj;
	  bool valid = pfTransformer_->addPoints( pfRecTrack, *trackRef, FakeTraj);
	  if(valid) 
	    {
	      double stip=-999;
	      const reco::PFTrajectoryPoint& atECAL=pfRecTrack.extrapolatedPoint(reco::PFTrajectoryPoint::ECALEntrance);
	      //if extrapolation to ECAL is valid then calculate STIP
	      if(atECAL.isValid())
		{
		  GlobalVector direction(pfRecTrack.extrapolatedPoint(reco::PFTrajectoryPoint::ECALEntrance).position().x(),
					 pfRecTrack.extrapolatedPoint(reco::PFTrajectoryPoint::ECALEntrance).position().y(), 
					 pfRecTrack.extrapolatedPoint(reco::PFTrajectoryPoint::ECALEntrance).position().z());
		  stip = IPTools::signedTransverseImpactParameter(thebuilder.build(*trackRef), direction, *pv).second.significance();
		}
	      pfRecTrack.setSTIP(stip);	    
	      pfRecTkcoll.push_back(reco::PFRecTrackRef( pfTrackRefProd, idx++));    	   
	      pfRecTrackColl->push_back(pfRecTrack);	    
	    }
	}//end loop over tracks
      //store reference to the Conversion collection
      reco::ConversionRef niRef(convCollH, i->first);
      pfConversionColl->push_back( reco::PFConversion( niRef, pfRecTkcoll ));
    }//end loop over collections

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
