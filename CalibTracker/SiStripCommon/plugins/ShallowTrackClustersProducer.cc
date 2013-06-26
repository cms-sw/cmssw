#include "CalibTracker/SiStripCommon/interface/ShallowTrackClustersProducer.h"

#include "CalibTracker/SiStripCommon/interface/ShallowTools.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
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

ShallowTrackClustersProducer::ShallowTrackClustersProducer(const edm::ParameterSet& iConfig)
  :  theTracksLabel( iConfig.getParameter<edm::InputTag>("Tracks") ),
     theClustersLabel( iConfig.getParameter<edm::InputTag>("Clusters") ),
     Suffix       ( iConfig.getParameter<std::string>("Suffix")    ),
     Prefix       ( iConfig.getParameter<std::string>("Prefix") )
{
  produces <std::vector<unsigned int> > ( Prefix + "trackmulti"  + Suffix );
  produces <std::vector<int> >          ( Prefix + "trackindex"  + Suffix );
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
  shallow::CLUSTERMAP clustermap = shallow::make_cluster_map(iEvent,theClustersLabel);

  int size = clustermap.size();
  std::auto_ptr<std::vector<unsigned int> > trackmulti   ( new std::vector<unsigned int>(size,    0)   );
  std::auto_ptr<std::vector<int> >          trackindex   ( new std::vector<int>         (size,   -1)   );
  std::auto_ptr<std::vector<float> >        localtheta   ( new std::vector<float>       (size, -100)   );
  std::auto_ptr<std::vector<float> >        localphi     ( new std::vector<float>       (size, -100)   );
  std::auto_ptr<std::vector<float> >        localpitch   ( new std::vector<float>       (size, -100)   );
  std::auto_ptr<std::vector<float> >        localx       ( new std::vector<float>       (size, -100)   );
  std::auto_ptr<std::vector<float> >        localy       ( new std::vector<float>       (size, -100)   );
  std::auto_ptr<std::vector<float> >        localz       ( new std::vector<float>       (size, -100)   );
  std::auto_ptr<std::vector<float> >        strip        ( new std::vector<float>       (size, -100)   );
  std::auto_ptr<std::vector<float> >        globaltheta  ( new std::vector<float>       (size, -100)   );
  std::auto_ptr<std::vector<float> >        globalphi    ( new std::vector<float>       (size, -100)   );
  std::auto_ptr<std::vector<float> >        globalx      ( new std::vector<float>       (size, -10000) );
  std::auto_ptr<std::vector<float> >        globaly      ( new std::vector<float>       (size, -10000) );
  std::auto_ptr<std::vector<float> >        globalz      ( new std::vector<float>       (size, -10000) );
  std::auto_ptr<std::vector<float> >        insidistance ( new std::vector<float>       (size,     -1) );
  std::auto_ptr<std::vector<float> >        projwidth    ( new std::vector<float>       (size,  -1000) );
  std::auto_ptr<std::vector<float> >        BdotY        ( new std::vector<float>       (size,  -1000) );
  std::auto_ptr<std::vector<float> >        covered      ( new std::vector<float>       (size,  -1000) );
  std::auto_ptr<std::vector<float> >  rhlocalx      ( new std::vector<float>(size,  -10000  ));   
  std::auto_ptr<std::vector<float> >  rhlocaly      ( new std::vector<float>(size,  -10000  ));   
  std::auto_ptr<std::vector<float> >  rhlocalxerr   ( new std::vector<float>(size,  -1  ));   
  std::auto_ptr<std::vector<float> >  rhlocalyerr   ( new std::vector<float>(size,  -1  ));     
  std::auto_ptr<std::vector<float> >  rhglobalx     ( new std::vector<float>(size,  -10000  ));   
  std::auto_ptr<std::vector<float> >  rhglobaly     ( new std::vector<float>(size,  -10000  ));   
  std::auto_ptr<std::vector<float> >  rhglobalz     ( new std::vector<float>(size,  -10000  ));   
  std::auto_ptr<std::vector<float> >  rhstrip       ( new std::vector<float>(size,  -10000  ));   
  std::auto_ptr<std::vector<float> >  rhmerr        ( new std::vector<float>(size,  -10000  ));   
  std::auto_ptr<std::vector<float> >  ubstrip       ( new std::vector<float>(size,  -10000  ));   
  std::auto_ptr<std::vector<float> >  ubmerr        ( new std::vector<float>(size,  -10000  ));   
  std::auto_ptr<std::vector<float> >  driftx        ( new std::vector<float>(size,  -10000  ));
  std::auto_ptr<std::vector<float> >  drifty        ( new std::vector<float>(size,  -10000  ));
  std::auto_ptr<std::vector<float> >  driftz        ( new std::vector<float>(size,  -10000  ));
  std::auto_ptr<std::vector<float> >  globalZofunitlocalY ( new std::vector<float>(size, -1000));

  edm::ESHandle<TrackerGeometry> theTrackerGeometry;         iSetup.get<TrackerDigiGeometryRecord>().get( theTrackerGeometry );  
  edm::ESHandle<MagneticField> magfield;		     iSetup.get<IdealMagneticFieldRecord>().get(magfield);		      
  edm::ESHandle<SiStripLorentzAngle> SiStripLorentzAngle;    iSetup.get<SiStripLorentzAngleDepRcd>().get(SiStripLorentzAngle);      

  edm::Handle<edm::View<reco::Track> > tracks;	             iEvent.getByLabel(theTracksLabel, tracks);	  
  edm::Handle<TrajTrackAssociationCollection> associations;  iEvent.getByLabel(theTracksLabel, associations);

  TrajectoryStateCombiner combiner;

  for( TrajTrackAssociationCollection::const_iterator association = associations->begin(); 
       association != associations->end(); association++) {
    const Trajectory*  traj  = association->key.get();
    const reco::Track* track = association->val.get();

    BOOST_FOREACH( const TrajectoryMeasurement measurement, traj->measurements() ) {
      const TrajectoryStateOnSurface tsos = measurement.updatedState();
      const TrajectoryStateOnSurface unbiased = combiner(measurement.forwardPredictedState(), measurement.backwardPredictedState());

      const TrackingRecHit*         hit        = measurement.recHit()->hit();
      const SiStripRecHit1D*        hit1D      = dynamic_cast<const SiStripRecHit1D*>(hit);
      const SiStripRecHit2D*        hit2D      = dynamic_cast<const SiStripRecHit2D*>(hit);
      const SiStripMatchedRecHit2D* matchedhit = dynamic_cast<const SiStripMatchedRecHit2D*>(hit);

      for(unsigned h=0; h<2; h++) {
	const SiStripCluster* cluster_ptr;
	if(!matchedhit && h==1) continue; else 
	if( matchedhit && h==0) cluster_ptr = &matchedhit->monoCluster(); else 
	if( matchedhit && h==1) cluster_ptr = &matchedhit->stereoCluster(); else 
	if(hit2D) cluster_ptr = (hit2D->cluster()).get(); else 
	if(hit1D) cluster_ptr = (hit1D->cluster()).get(); 
	else continue;

	shallow::CLUSTERMAP::const_iterator cluster = clustermap.find( std::make_pair( hit->geographicalId().rawId(), cluster_ptr->firstStrip() ));
	if(cluster == clustermap.end() ) throw cms::Exception("Logic Error") << "Cluster not found: this could be a configuration error" << std::endl;
	
	unsigned i = cluster->second;
	if( 0 == (trackmulti->at(i))++ ) {
	  const StripGeomDetUnit* theStripDet = dynamic_cast<const StripGeomDetUnit*>( theTrackerGeometry->idToDet( hit->geographicalId() ) );
	  LocalVector drift = shallow::drift( theStripDet, *magfield, *SiStripLorentzAngle);
	  
	  trackindex->at(i)   = shallow::findTrackIndex(tracks, track); 
	  localtheta->at(i)   = (theStripDet->toLocal(tsos.globalDirection())).theta(); 
	  localphi->at(i)     = (theStripDet->toLocal(tsos.globalDirection())).phi();   
	  localpitch->at(i)   = (theStripDet->specificTopology()).localPitch(theStripDet->toLocal(tsos.globalPosition())); 
	  localx->at(i)       = (theStripDet->toLocal(tsos.globalPosition())).x();    
	  localy->at(i)       = (theStripDet->toLocal(tsos.globalPosition())).y();    
	  localz->at(i)       = (theStripDet->toLocal(tsos.globalPosition())).z();    
	  strip->at(i)        = (theStripDet->specificTopology()).strip(theStripDet->toLocal(tsos.globalPosition()));
	  globaltheta->at(i)  = tsos.globalDirection().theta();                       
	  globalphi->at(i)    = tsos.globalDirection().phi();                         
	  globalx->at(i)      = tsos.globalPosition().x();                            
	  globaly->at(i)      = tsos.globalPosition().y();                            
	  globalz->at(i)      = tsos.globalPosition().z();                            
	  insidistance->at(i) = 1./fabs(cos(localtheta->at(i)));                      
	  projwidth->at(i)    = tan(localtheta->at(i))*cos(localphi->at(i));         
	  BdotY->at(i)        = (theStripDet->surface()).toLocal( magfield->inTesla(theStripDet->surface().position())).y();
	  covered->at(i)      = drift.z()/localpitch->at(i) * fabs(projwidth->at(i) - drift.x()/drift.z());
	  rhlocalx->at(i)     = hit->localPosition().x();
	  rhlocaly->at(i)     = hit->localPosition().y();
	  rhlocalxerr->at(i)  = sqrt(hit->localPositionError().xx());
	  rhlocalyerr->at(i)  = sqrt(hit->localPositionError().yy());
	  rhglobalx->at(i)    = theStripDet->toGlobal(hit->localPosition()).x();
	  rhglobaly->at(i)    = theStripDet->toGlobal(hit->localPosition()).y();
	  rhglobalz->at(i)    = theStripDet->toGlobal(hit->localPosition()).z();
	  rhstrip->at(i)      = theStripDet->specificTopology().strip(hit->localPosition());
	  rhmerr->at(i)       = sqrt(theStripDet->specificTopology().measurementError(hit->localPosition(), hit->localPositionError()).uu());
	  ubstrip->at(i)      = theStripDet->specificTopology().strip(unbiased.localPosition());
	  ubmerr->at(i)       = sqrt(theStripDet->specificTopology().measurementError(unbiased.localPosition(), unbiased.localError().positionError()).uu());
	  driftx->at(i)       = drift.x();
	  drifty->at(i)       = drift.y();
	  driftz->at(i)       = drift.z();
	  globalZofunitlocalY->at(i) = (theStripDet->toGlobal(LocalVector(0,1,0))).z();
	}
      }
    }
  }

  iEvent.put(trackmulti,  Prefix + "trackmulti"  + Suffix );
  iEvent.put(trackindex,  Prefix + "trackindex"  + Suffix );
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
