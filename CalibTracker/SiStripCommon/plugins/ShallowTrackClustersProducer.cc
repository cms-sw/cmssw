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
  auto clusterIdx        = std::make_unique<std::vector<int>>(); //link: on trk cluster --> general cluster info 
  auto onTrkClusterIdx   = std::make_unique<std::vector<int>>(size,   -1); //link: general cluster info --> on track cluster
  auto onTrkClustersBegin = std::make_unique<std::vector<int>>( tracks->size(), -1 ); //link: track --> on trk cluster
  auto onTrkClustersEnd   = std::make_unique<std::vector<int>>( tracks->size(), -1 ); //link: track --> on trk cluster
  auto trackindex    = std::make_unique<std::vector<int>>(); //link: on track cluster --> track
	clusterIdx->reserve(size);
	trackindex->reserve(size);
	
  auto  trackmulti   = std::make_unique<std::vector<unsigned int>>(); trackmulti  ->reserve(size);
  auto  localtheta   = std::make_unique<std::vector<float>>(); localtheta  ->reserve(size);
  auto  localphi     = std::make_unique<std::vector<float>>(); localphi    ->reserve(size);
  auto  localpitch   = std::make_unique<std::vector<float>>(); localpitch  ->reserve(size);
  auto  localx       = std::make_unique<std::vector<float>>(); localx      ->reserve(size);
  auto  localy       = std::make_unique<std::vector<float>>(); localy      ->reserve(size);
  auto  localz       = std::make_unique<std::vector<float>>(); localz      ->reserve(size);
  auto  strip        = std::make_unique<std::vector<float>>(); strip       ->reserve(size);
  auto  globaltheta  = std::make_unique<std::vector<float>>(); globaltheta ->reserve(size);
  auto  globalphi    = std::make_unique<std::vector<float>>(); globalphi   ->reserve(size);
  auto  globalx      = std::make_unique<std::vector<float>>(); globalx     ->reserve(size);
  auto  globaly      = std::make_unique<std::vector<float>>(); globaly     ->reserve(size);
  auto  globalz      = std::make_unique<std::vector<float>>(); globalz     ->reserve(size);
  auto  insidistance = std::make_unique<std::vector<float>>(); insidistance->reserve(size);
  auto  projwidth    = std::make_unique<std::vector<float>>(); projwidth   ->reserve(size);
  auto  BdotY        = std::make_unique<std::vector<float>>(); BdotY       ->reserve(size);
  auto  covered      = std::make_unique<std::vector<float>>(); covered     ->reserve(size);
  auto  rhlocalx     = std::make_unique<std::vector<float>>(); rhlocalx   ->reserve(size);
  auto  rhlocaly     = std::make_unique<std::vector<float>>(); rhlocaly   ->reserve(size);  
  auto  rhlocalxerr  = std::make_unique<std::vector<float>>(); rhlocalxerr->reserve(size);  
  auto  rhlocalyerr  = std::make_unique<std::vector<float>>(); rhlocalyerr->reserve(size);    
  auto  rhglobalx    = std::make_unique<std::vector<float>>(); rhglobalx  ->reserve(size);  
  auto  rhglobaly    = std::make_unique<std::vector<float>>(); rhglobaly  ->reserve(size);  
  auto  rhglobalz    = std::make_unique<std::vector<float>>(); rhglobalz  ->reserve(size);  
  auto  rhstrip      = std::make_unique<std::vector<float>>(); rhstrip    ->reserve(size);  
  auto  rhmerr       = std::make_unique<std::vector<float>>(); rhmerr     ->reserve(size);  
  auto  ubstrip      = std::make_unique<std::vector<float>>(); ubstrip    ->reserve(size);  
  auto  ubmerr       = std::make_unique<std::vector<float>>(); ubmerr     ->reserve(size);  
  auto  driftx       = std::make_unique<std::vector<float>>(); driftx     ->reserve(size);
  auto  drifty       = std::make_unique<std::vector<float>>(); drifty     ->reserve(size);
  auto  driftz       = std::make_unique<std::vector<float>>(); driftz     ->reserve(size);
  auto  globalZofunitlocalY = std::make_unique<std::vector<float>>(); globalZofunitlocalY->reserve(size);

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

    for(auto const& measurement : traj->measurements() ) {
      const TrajectoryStateOnSurface& tsos = measurement.updatedState();
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
    } //for(auto const& measurement : traj->measurements() )

		onTrkClustersBegin->at(trk_idx) = trk_strt_idx;
		onTrkClustersEnd->at(trk_idx)   = ontrk_cluster_idx;

  } //for(TrajTrackAssociationCollection::const_iterator association = associations->begin();

  iEvent.put(std::move(clusterIdx        ), Prefix + "clusterIdx" + Suffix );
  iEvent.put(std::move(onTrkClusterIdx   ), Prefix + "onTrkClusterIdx" + Suffix );
  iEvent.put(std::move(onTrkClustersBegin), Prefix + "onTrkClustersBegin" + Suffix );
  iEvent.put(std::move(onTrkClustersEnd  ), Prefix + "onTrkClustersEnd" + Suffix );
  iEvent.put(std::move(trackindex),  Prefix + "trackindex"  + Suffix );

  iEvent.put(std::move(trackmulti),  Prefix + "trackmulti"  + Suffix );
  iEvent.put(std::move(localtheta),  Prefix + "localtheta"  + Suffix );
  iEvent.put(std::move(localphi),    Prefix + "localphi"    + Suffix );
  iEvent.put(std::move(localpitch),  Prefix + "localpitch"  + Suffix );
  iEvent.put(std::move(localx),      Prefix + "localx"      + Suffix );
  iEvent.put(std::move(localy),      Prefix + "localy"      + Suffix );
  iEvent.put(std::move(localz),      Prefix + "localz"      + Suffix );
  iEvent.put(std::move(strip),       Prefix + "strip"       + Suffix );
  iEvent.put(std::move(globaltheta), Prefix + "globaltheta" + Suffix );
  iEvent.put(std::move(globalphi),   Prefix + "globalphi"   + Suffix );
  iEvent.put(std::move(globalx),     Prefix + "globalx"     + Suffix );
  iEvent.put(std::move(globaly),     Prefix + "globaly"     + Suffix );
  iEvent.put(std::move(globalz),     Prefix + "globalz"     + Suffix );  
  iEvent.put(std::move(insidistance),Prefix + "insidistance"+ Suffix );
  iEvent.put(std::move(covered),     Prefix + "covered"     + Suffix );
  iEvent.put(std::move(projwidth),   Prefix + "projwidth"   + Suffix );
  iEvent.put(std::move(BdotY),       Prefix + "BdotY"       + Suffix );
  iEvent.put(std::move(rhlocalx),    Prefix + "rhlocalx"    + Suffix );   
  iEvent.put(std::move(rhlocaly),    Prefix + "rhlocaly"    + Suffix );   
  iEvent.put(std::move(rhlocalxerr), Prefix + "rhlocalxerr" + Suffix );   
  iEvent.put(std::move(rhlocalyerr), Prefix + "rhlocalyerr" + Suffix );   
  iEvent.put(std::move(rhglobalx),   Prefix + "rhglobalx"   + Suffix );   
  iEvent.put(std::move(rhglobaly),   Prefix + "rhglobaly"   + Suffix );   
  iEvent.put(std::move(rhglobalz),   Prefix + "rhglobalz"   + Suffix );   
  iEvent.put(std::move(rhstrip),     Prefix + "rhstrip"     + Suffix );   
  iEvent.put(std::move(rhmerr),      Prefix + "rhmerr"      + Suffix );   
  iEvent.put(std::move(ubstrip),     Prefix + "ubstrip"     + Suffix );   
  iEvent.put(std::move(ubmerr),      Prefix + "ubmerr"      + Suffix );   
  iEvent.put(std::move(driftx),      Prefix + "driftx"      + Suffix );
  iEvent.put(std::move(drifty),      Prefix + "drifty"      + Suffix );
  iEvent.put(std::move(driftz),      Prefix + "driftz"      + Suffix );
  iEvent.put(std::move(globalZofunitlocalY), Prefix + "globalZofunitlocalY" + Suffix );
}
