#include "CalibTracker/SiStripCommon/interface/ShallowTrackClustersProducer.h"

#include "CalibTracker/SiStripCommon/interface/ShallowTools.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "CondFormats/DataRecord/interface/SiStripLorentzAngleRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "boost/foreach.hpp"

#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include <map>

ShallowTrackClustersProducer::ShallowTrackClustersProducer(const edm::ParameterSet& iConfig)
  :  tracks_token_(consumes<edm::View<reco::Track> >(iConfig.getParameter<edm::InputTag>("Tracks"))),
		 association_token_(consumes<TrajTrackAssociationCollection>(iConfig.getParameter<edm::InputTag>("Tracks"))),
     clusters_token_( consumes< edmNew::DetSetVector<SiStripCluster> >( iConfig.getParameter<edm::InputTag>("Clusters") ) ),
     Suffix       ( iConfig.getParameter<std::string>("Suffix")    ),
     Prefix       ( iConfig.getParameter<std::string>("Prefix") )
{
  produces<std::vector<int> > ( Prefix + "clusterIdx"      + Suffix ); //link: on trk cluster --> general cluster info 
  produces<std::vector<int> > ( Prefix + "onTrkClusterIdx" + Suffix ); //link: general cluster info --> on track cluster
	produces <std::vector<int> > ( Prefix + "onTrkClustersBegin" + Suffix ); //link: track --> onTrkInfo (range)
	produces <std::vector<int> > ( Prefix + "onTrkClustersEnd" + Suffix ); //link: track --> onTrkInfo (range)
  produces <std::vector<int> > ( Prefix + "trackindex"  + Suffix ); //link: on trk cluster --> track index

  produces <std::vector<unsigned int> > ( Prefix + "trackmulti"  + Suffix );
  produces <std::vector<float> >        ( Prefix + "localtheta"  + Suffix );  
  produces <std::vector<float> >        ( Prefix + "localphi"    + Suffix );  
  produces <std::vector<float> >        ( Prefix + "localpitch"  + Suffix );  
  produces <std::vector<float> >        ( Prefix + "localx"      + Suffix );  
  produces <std::vector<float> >        ( Prefix + "localy"      + Suffix );  
  produces <std::vector<float> >        ( Prefix + "localz"      + Suffix );  
  produces <std::vector<float> >        ( Prefix + "strip"       + Suffix );  
  produces <std::vector<float> >        ( Prefix + "globaltheta" + Suffix );  
  produces <std::vector<float> >        ( Prefix + "globalphi"   + Suffix );
  produces <std::vector<float> >        ( Prefix + "globalx"     + Suffix );
  produces <std::vector<float> >        ( Prefix + "globaly"     + Suffix );
  produces <std::vector<float> >        ( Prefix + "globalz"     + Suffix );
  produces <std::vector<float> >        ( Prefix + "insidistance"+ Suffix );
  produces <std::vector<float> >        ( Prefix + "covered"     + Suffix );
  produces <std::vector<float> >        ( Prefix + "projwidth"   + Suffix );
  produces <std::vector<float> >        ( Prefix + "BdotY"       + Suffix );

  produces <std::vector<float> >        ( Prefix + "rhlocalx"     + Suffix );   
  produces <std::vector<float> >        ( Prefix + "rhlocaly"     + Suffix );   
  produces <std::vector<float> >        ( Prefix + "rhlocalxerr"  + Suffix );   
  produces <std::vector<float> >        ( Prefix + "rhlocalyerr"  + Suffix );   
  produces <std::vector<float> >        ( Prefix + "rhglobalx"    + Suffix );   
  produces <std::vector<float> >        ( Prefix + "rhglobaly"    + Suffix );   
  produces <std::vector<float> >        ( Prefix + "rhglobalz"    + Suffix );
  produces <std::vector<float> >        ( Prefix + "rhstrip"      + Suffix );   
  produces <std::vector<float> >        ( Prefix + "rhmerr"       + Suffix );   

  produces <std::vector<float> >        ( Prefix + "ubstrip"      + Suffix );   
  produces <std::vector<float> >        ( Prefix + "ubmerr"       + Suffix );   

  produces <std::vector<float> >       ( Prefix + "driftx"        + Suffix );
  produces <std::vector<float> >       ( Prefix + "drifty"        + Suffix );
  produces <std::vector<float> >       ( Prefix + "driftz"        + Suffix );
  produces <std::vector<float> >       ( Prefix + "globalZofunitlocalY" + Suffix );            
}

void ShallowTrackClustersProducer::
produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  shallow::CLUSTERMAP clustermap = shallow::make_cluster_map(iEvent, clusters_token_);
  edm::Handle<edm::View<reco::Track> > tracks;	             iEvent.getByToken(tracks_token_, tracks);	  

  int size = clustermap.size();

	//links
  std::auto_ptr<std::vector<int> > clusterIdx        ( new std::vector<int>() ); //link: on trk cluster --> general cluster info 
  std::auto_ptr<std::vector<int> > onTrkClusterIdx   ( new std::vector<int>(size,   -1)   ); //link: general cluster info --> on track cluster
	std::auto_ptr<std::vector<int> > onTrkClustersBegin (new std::vector<int>( tracks->size(), -1 ) ); //link: track --> on trk cluster
	std::auto_ptr<std::vector<int> > onTrkClustersEnd   (new std::vector<int>( tracks->size(), -1 ) ); //link: track --> on trk cluster
  std::auto_ptr<std::vector<int> > trackindex    ( new std::vector<int>()   ); //link: on track cluster --> track
	clusterIdx->reserve(size);
	trackindex->reserve(size);
	
  std::auto_ptr<std::vector<unsigned int> > trackmulti   ( new std::vector<unsigned int>()); trackmulti  ->reserve(size);
  std::auto_ptr<std::vector<float> >        localtheta   ( new std::vector<float>       ()); localtheta  ->reserve(size);
  std::auto_ptr<std::vector<float> >        localphi     ( new std::vector<float>       ()); localphi    ->reserve(size);
  std::auto_ptr<std::vector<float> >        localpitch   ( new std::vector<float>       ()); localpitch  ->reserve(size);
  std::auto_ptr<std::vector<float> >        localx       ( new std::vector<float>       ()); localx      ->reserve(size);
  std::auto_ptr<std::vector<float> >        localy       ( new std::vector<float>       ()); localy      ->reserve(size);
  std::auto_ptr<std::vector<float> >        localz       ( new std::vector<float>       ()); localz      ->reserve(size);
  std::auto_ptr<std::vector<float> >        strip        ( new std::vector<float>       ()); strip       ->reserve(size);
  std::auto_ptr<std::vector<float> >        globaltheta  ( new std::vector<float>       ()); globaltheta ->reserve(size);
  std::auto_ptr<std::vector<float> >        globalphi    ( new std::vector<float>       ()); globalphi   ->reserve(size);
  std::auto_ptr<std::vector<float> >        globalx      ( new std::vector<float>       ()); globalx     ->reserve(size);
  std::auto_ptr<std::vector<float> >        globaly      ( new std::vector<float>       ()); globaly     ->reserve(size);
  std::auto_ptr<std::vector<float> >        globalz      ( new std::vector<float>       ()); globalz     ->reserve(size);
  std::auto_ptr<std::vector<float> >        insidistance ( new std::vector<float>       ()); insidistance->reserve(size);
  std::auto_ptr<std::vector<float> >        projwidth    ( new std::vector<float>       ()); projwidth   ->reserve(size);
  std::auto_ptr<std::vector<float> >        BdotY        ( new std::vector<float>       ()); BdotY       ->reserve(size);
  std::auto_ptr<std::vector<float> >        covered      ( new std::vector<float>       ()); covered     ->reserve(size);
  std::auto_ptr<std::vector<float> >  rhlocalx   ( new std::vector<float>()); rhlocalx   ->reserve(size);
  std::auto_ptr<std::vector<float> >  rhlocaly   ( new std::vector<float>()); rhlocaly   ->reserve(size);  
  std::auto_ptr<std::vector<float> >  rhlocalxerr( new std::vector<float>()); rhlocalxerr->reserve(size);  
  std::auto_ptr<std::vector<float> >  rhlocalyerr( new std::vector<float>()); rhlocalyerr->reserve(size);    
  std::auto_ptr<std::vector<float> >  rhglobalx  ( new std::vector<float>()); rhglobalx  ->reserve(size);  
  std::auto_ptr<std::vector<float> >  rhglobaly  ( new std::vector<float>()); rhglobaly  ->reserve(size);  
  std::auto_ptr<std::vector<float> >  rhglobalz  ( new std::vector<float>()); rhglobalz  ->reserve(size);  
  std::auto_ptr<std::vector<float> >  rhstrip    ( new std::vector<float>()); rhstrip    ->reserve(size);  
  std::auto_ptr<std::vector<float> >  rhmerr     ( new std::vector<float>()); rhmerr     ->reserve(size);  
  std::auto_ptr<std::vector<float> >  ubstrip    ( new std::vector<float>()); ubstrip    ->reserve(size);  
  std::auto_ptr<std::vector<float> >  ubmerr     ( new std::vector<float>()); ubmerr     ->reserve(size);  
  std::auto_ptr<std::vector<float> >  driftx     ( new std::vector<float>());	driftx     ->reserve(size);
  std::auto_ptr<std::vector<float> >  drifty     ( new std::vector<float>());	drifty     ->reserve(size);
  std::auto_ptr<std::vector<float> >  driftz     ( new std::vector<float>());	driftz     ->reserve(size);
  std::auto_ptr<std::vector<float> >  globalZofunitlocalY ( new std::vector<float>()); globalZofunitlocalY->reserve(size);

  edm::ESHandle<TrackerGeometry> theTrackerGeometry;         iSetup.get<TrackerDigiGeometryRecord>().get( theTrackerGeometry );  
  edm::ESHandle<MagneticField> magfield;		     iSetup.get<IdealMagneticFieldRecord>().get(magfield);		      
  edm::ESHandle<SiStripLorentzAngle> SiStripLorentzAngle;    iSetup.get<SiStripLorentzAngleDepRcd>().get(SiStripLorentzAngle);      

  edm::Handle<TrajTrackAssociationCollection> associations;  iEvent.getByToken(association_token_, associations);

  TrajectoryStateCombiner combiner;

	size_t ontrk_cluster_idx=0;
  std::map<size_t, std::vector<size_t> > mapping; //cluster idx --> on trk cluster idx (multiple)

  for( TrajTrackAssociationCollection::const_iterator association = associations->begin(); 
       association != associations->end(); association++) {
    const Trajectory*  traj  = association->key.get();
    const reco::Track* track = association->val.get();
		int trk_idx = shallow::findTrackIndex(tracks, track); 
		size_t trk_strt_idx = ontrk_cluster_idx;

    BOOST_FOREACH( const TrajectoryMeasurement measurement, traj->measurements() ) {
      const TrajectoryStateOnSurface tsos = measurement.updatedState();
      const TrajectoryStateOnSurface unbiased = combiner(measurement.forwardPredictedState(), measurement.backwardPredictedState());

      const TrackingRecHit*         hit        = measurement.recHit()->hit();
      const SiStripRecHit1D*        hit1D      = dynamic_cast<const SiStripRecHit1D*>(hit);
      const SiStripRecHit2D*        hit2D      = dynamic_cast<const SiStripRecHit2D*>(hit);
      const SiStripMatchedRecHit2D* matchedhit = dynamic_cast<const SiStripMatchedRecHit2D*>(hit);

      for(unsigned h=0; h<2; h++) { //loop over possible Hit options (1D, 2D)
				const SiStripCluster* cluster_ptr;
				if(!matchedhit && h==1) continue; 
				else if( matchedhit && h==0) cluster_ptr = &matchedhit->monoCluster(); 
				else if( matchedhit && h==1) cluster_ptr = &matchedhit->stereoCluster(); 
				else if(hit2D) cluster_ptr = (hit2D->cluster()).get(); 
				else if(hit1D) cluster_ptr = (hit1D->cluster()).get(); 
				else continue;

				shallow::CLUSTERMAP::const_iterator cluster = clustermap.find( std::make_pair( hit->geographicalId().rawId(), cluster_ptr->firstStrip() ));
				if(cluster == clustermap.end() ) throw cms::Exception("Logic Error") << "Cluster not found: this could be a configuration error" << std::endl;
	
				unsigned i = cluster->second;

				//find if cluster was already assigned to a previous track
				auto already_visited = mapping.find(i);
				int nassociations = 1;
				if(already_visited != mapping.end()) {
					nassociations += already_visited->second.size();
					for(size_t idx : already_visited->second) {
						trackmulti->at(idx)++;
					}
					already_visited->second.push_back(ontrk_cluster_idx);
				}
				else { //otherwise store this 
					std::vector<size_t> single = {ontrk_cluster_idx};
					mapping.insert( std::make_pair(i, single) );
				}

				const StripGeomDetUnit* theStripDet = dynamic_cast<const StripGeomDetUnit*>( theTrackerGeometry->idToDet( hit->geographicalId() ) );
				LocalVector drift = shallow::drift( theStripDet, *magfield, *SiStripLorentzAngle);
				
				if(nassociations == 1) onTrkClusterIdx->at(i) = ontrk_cluster_idx; //link: general cluster info --> on track cluster
				clusterIdx->push_back(  i );  //link: on trk cluster --> general cluster info 
				trackmulti->push_back(  nassociations );
				trackindex->push_back(  trk_idx );
				localtheta->push_back(  (theStripDet->toLocal(tsos.globalDirection())).theta() ); 
				localphi->push_back(    (theStripDet->toLocal(tsos.globalDirection())).phi() );   
				localpitch->push_back(  (theStripDet->specificTopology()).localPitch(theStripDet->toLocal(tsos.globalPosition())) ); 
				localx->push_back(      (theStripDet->toLocal(tsos.globalPosition())).x() );    
				localy->push_back(      (theStripDet->toLocal(tsos.globalPosition())).y() );    
				localz->push_back(      (theStripDet->toLocal(tsos.globalPosition())).z() );    
				strip->push_back(       (theStripDet->specificTopology()).strip(theStripDet->toLocal(tsos.globalPosition())) );
				globaltheta->push_back( tsos.globalDirection().theta() );                       
				globalphi->push_back(   tsos.globalDirection().phi() );                         
				globalx->push_back(     tsos.globalPosition().x() );                            
				globaly->push_back(     tsos.globalPosition().y() );                            
				globalz->push_back(     tsos.globalPosition().z() );                            
				insidistance->push_back(1./fabs(cos(localtheta->at(ontrk_cluster_idx))) );                      
				projwidth->push_back(   tan(localtheta->at(ontrk_cluster_idx))*cos(localphi->at(ontrk_cluster_idx)) );         
				BdotY->push_back(       (theStripDet->surface()).toLocal( magfield->inTesla(theStripDet->surface().position())).y() );
				covered->push_back(     drift.z()/localpitch->at(ontrk_cluster_idx) * fabs(projwidth->at(ontrk_cluster_idx) - drift.x()/drift.z()) );
				rhlocalx->push_back(    hit->localPosition().x() );
				rhlocaly->push_back(    hit->localPosition().y() );
				rhlocalxerr->push_back( sqrt(hit->localPositionError().xx()) );
				rhlocalyerr->push_back( sqrt(hit->localPositionError().yy()) );
				rhglobalx->push_back(   theStripDet->toGlobal(hit->localPosition()).x() );
				rhglobaly->push_back(   theStripDet->toGlobal(hit->localPosition()).y() );
				rhglobalz->push_back(   theStripDet->toGlobal(hit->localPosition()).z() );
				rhstrip->push_back(     theStripDet->specificTopology().strip(hit->localPosition()) );
				rhmerr->push_back(      sqrt(theStripDet->specificTopology().measurementError(hit->localPosition(), hit->localPositionError()).uu()) );
				ubstrip->push_back(     theStripDet->specificTopology().strip(unbiased.localPosition()) );
				ubmerr->push_back(      sqrt(theStripDet->specificTopology().measurementError(unbiased.localPosition(), unbiased.localError().positionError()).uu()) );
				driftx->push_back(      drift.x() );
				drifty->push_back(      drift.y() );
				driftz->push_back(      drift.z() );
				globalZofunitlocalY->push_back( (theStripDet->toGlobal(LocalVector(0,1,0))).z() );
				
				ontrk_cluster_idx++;
      } //for(unsigned h=0; h<2; h++) { //loop over possible Hit options (1D, 2D)
    } //BOOST_FOREACH( const TrajectoryMeasurement measurement, traj->measurements() )

		onTrkClustersBegin->at(trk_idx) = trk_strt_idx;
		onTrkClustersEnd->at(trk_idx)   = ontrk_cluster_idx;

  } //for(TrajTrackAssociationCollection::const_iterator association = associations->begin();

  iEvent.put(clusterIdx        , Prefix + "clusterIdx" + Suffix );
  iEvent.put(onTrkClusterIdx   , Prefix + "onTrkClusterIdx" + Suffix );
	iEvent.put(onTrkClustersBegin, Prefix + "onTrkClustersBegin" + Suffix );
	iEvent.put(onTrkClustersEnd  , Prefix + "onTrkClustersEnd" + Suffix );
  iEvent.put(trackindex,  Prefix + "trackindex"  + Suffix );

  iEvent.put(trackmulti,  Prefix + "trackmulti"  + Suffix );
  iEvent.put(localtheta,  Prefix + "localtheta"  + Suffix );
  iEvent.put(localphi,    Prefix + "localphi"    + Suffix );
  iEvent.put(localpitch,  Prefix + "localpitch"  + Suffix );
  iEvent.put(localx,      Prefix + "localx"      + Suffix );
  iEvent.put(localy,      Prefix + "localy"      + Suffix );
  iEvent.put(localz,      Prefix + "localz"      + Suffix );
  iEvent.put(strip,       Prefix + "strip"       + Suffix );
  iEvent.put(globaltheta, Prefix + "globaltheta" + Suffix );
  iEvent.put(globalphi,   Prefix + "globalphi"   + Suffix );
  iEvent.put(globalx,     Prefix + "globalx"     + Suffix );
  iEvent.put(globaly,     Prefix + "globaly"     + Suffix );
  iEvent.put(globalz,     Prefix + "globalz"     + Suffix );  
  iEvent.put(insidistance,Prefix + "insidistance"+ Suffix );
  iEvent.put(covered,     Prefix + "covered"     + Suffix );
  iEvent.put(projwidth,   Prefix + "projwidth"   + Suffix );
  iEvent.put(BdotY,       Prefix + "BdotY"       + Suffix );
  iEvent.put(rhlocalx,    Prefix + "rhlocalx"    + Suffix );   
  iEvent.put(rhlocaly,    Prefix + "rhlocaly"    + Suffix );   
  iEvent.put(rhlocalxerr, Prefix + "rhlocalxerr" + Suffix );   
  iEvent.put(rhlocalyerr, Prefix + "rhlocalyerr" + Suffix );   
  iEvent.put(rhglobalx,   Prefix + "rhglobalx"   + Suffix );   
  iEvent.put(rhglobaly,   Prefix + "rhglobaly"   + Suffix );   
  iEvent.put(rhglobalz,   Prefix + "rhglobalz"   + Suffix );   
  iEvent.put(rhstrip,     Prefix + "rhstrip"     + Suffix );   
  iEvent.put(rhmerr,      Prefix + "rhmerr"      + Suffix );   
  iEvent.put(ubstrip,     Prefix + "ubstrip"     + Suffix );   
  iEvent.put(ubmerr,      Prefix + "ubmerr"      + Suffix );   
  iEvent.put( driftx,     Prefix + "driftx"      + Suffix );
  iEvent.put( drifty,     Prefix + "drifty"      + Suffix );
  iEvent.put( driftz,     Prefix + "driftz"      + Suffix );
  iEvent.put( globalZofunitlocalY, Prefix + "globalZofunitlocalY" + Suffix );
}
