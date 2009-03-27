/** \class MuonMillepedeTrackRefitter
 *  
 *
 *  $Date: 2008/12/12 18:02:14 $
 *  $Revision: 1.1 $
 *  \author P. Martinez Ruiz del Arbol, IFCA (CSIC-UC)  <Pablo.Martinez@cern.ch>
 */

#include "Alignment/MuonAlignmentAlgorithms/plugins/MuonMillepedeTrackRefitter.h"

// Collaborating Class Header
#include "Alignment/MuonAlignmentAlgorithms/interface/SegmentToTrackAssociator.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/Cone.h"

#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHitBuilder.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"


#include <vector>
#include "TH2D.h"

using namespace std;
using namespace edm;




/// Constructor
MuonMillepedeTrackRefitter::MuonMillepedeTrackRefitter( const ParameterSet& pset )
{
  
  //Parameters
  MuonCollectionTag = pset.getParameter<edm::InputTag>( "MuonCollectionTag" );
  
  TrackerCollectionTag = pset.getParameter<edm::InputTag>( "TrackerTrackCollectionTag" );
  
  SACollectionTag = pset.getParameter<edm::InputTag>( "SATrackCollectionTag" );
 
 
  propagatorSourceOpposite = pset.getParameter<std::string>( "PropagatorSourceOpposite" );
  
  propagatorSourceAlong = pset.getParameter<std::string>( "PropagatorSourceAlong" );

 
  const ParameterSet SegmentsTrackAssociatorParameters =
    pset.getParameter<ParameterSet>( "SegmentToTrackAssociatorParameters" );
    
   //Creation of SegmentToTrackAssociator
  theSegmentsAssociator = new SegmentToTrackAssociator( SegmentsTrackAssociatorParameters );
  
  //Products
  produces<vector<Trajectory> >();
  produces<TrajTrackAssociationCollection>();

  

}

// Destructor
MuonMillepedeTrackRefitter::~MuonMillepedeTrackRefitter()
{
}


void MuonMillepedeTrackRefitter::produce( Event & event, const EventSetup& eventSetup )
{
 
  //Get collections from the event

  edm::Handle<reco::MuonCollection> muons;
  event.getByLabel( MuonCollectionTag, muons );
  
  edm::Handle<reco::TrackCollection> tracksTR;
  event.getByLabel( TrackerCollectionTag, tracksTR );
  
  edm::Handle<reco::TrackCollection> tracksSA;
  event.getByLabel( SACollectionTag, tracksSA );

  edm::ESHandle<Propagator> thePropagatorOpp;
  eventSetup.get<TrackingComponentsRecord>().get( propagatorSourceOpposite, thePropagatorOpp );
  
  edm::ESHandle<Propagator> thePropagatorAlo;
  eventSetup.get<TrackingComponentsRecord>().get( propagatorSourceAlong, thePropagatorAlo );

  ESHandle<MagneticField> theMGField;
  eventSetup.get<IdealMagneticFieldRecord>().get( theMGField );

  RefProd<vector<Trajectory> > trajectoryCollectionRefProd 
    = event.getRefBeforePut<vector<Trajectory> >();

  ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  eventSetup.get<GlobalTrackingGeometryRecord>().get( theTrackingGeometry );

  //Allocate collection of tracks
  auto_ptr<vector<Trajectory> > trajectoryCollection( new vector<Trajectory> );
  // Association map between Trajectory and Track
  auto_ptr<TrajTrackAssociationCollection> trajTrackMap( new TrajTrackAssociationCollection );
 

  //Create the propagator
  //thePropagator = new SteppingHelixPropagator( &*theMGField, alongMomentum );
    
  theSegmentsAssociator->clear();
 
  std::map<edm::Ref<std::vector<Trajectory> >::key_type, edm::Ref<reco::TrackCollection>::key_type> trajToTrack_map;
  
  Ref<vector<Trajectory> >::key_type trajectoryIndex = 0;

     
     for ( reco::MuonCollection::const_iterator muon = muons->begin();  muon != muons->end();  ++muon ) {
   
    reco::TrackRef::key_type trackIndex = 0;

    //reco::MuonCollection::const_iterator muon = muons->begin();

    if( muon->isGlobalMuon() )
    {
     
      reco::TrackRef trackTR = muon->innerTrack();
      reco::TrackRef trackSA = muon->outerTrack();
      reco::TrackRef trackGR = muon->combinedMuon();
  
      reco::TrackCollection::const_iterator trackSAIt = tracksSA->begin(); 
      reco::TrackCollection::const_iterator trackTRIt = tracksTR->begin(); 


      for ( ; trackTRIt != tracksTR->end();  ++trackTRIt, ++trackSAIt )
      {
        if ( fabs(trackTRIt->phi() - trackTR->phi()) < 1e-6 && fabs(trackTRIt->theta() - trackTR->theta()) < 1e-6 && 
             fabs(trackSAIt->phi() - trackSA->phi()) < 1e-6 && fabs(trackSAIt->theta() - trackSA->theta()) < 1e-6 ) 
          break;
        ++trackIndex;
        if (trackTRIt == tracksTR->end()) {
          throw cms::Exception("BadConfig") << "Muon Collection is not refering the proper TrackerCollection " << std::endl;
        }
      }
      
      //Get a transient Track for the Tracker
      reco::TransientTrack tTrackTR( *trackTRIt, &*theMGField, theTrackingGeometry );
      
      reco::TransientTrack tTrackSA( *trackSAIt, &*theMGField, theTrackingGeometry );

      reco::TransientTrack tTrackGR( *trackGR,  &*theMGField, theTrackingGeometry );

      //Create an empty trajectory
      Trajectory myTraj;
    
   
      // Adapted code for muonCosmics
   
      Double_t  innerPerpSA  = tTrackSA.innermostMeasurementState().globalPosition().perp();
      Double_t  outerPerpSA  = tTrackSA.outermostMeasurementState().globalPosition().perp();
     
      TrajectoryStateOnSurface innerTSOS=tTrackTR.outermostMeasurementState();
      //PropagationDirection propagationDir;
   
      const Propagator *thePropagator;      

      edm::LogWarning("Alignment") << "TestProp New track" << std::endl; 

      // Define which kind of reco track is used
      if ( (outerPerpSA-innerPerpSA) > 0 )
      {
	TrackRefitterType = "LHCLike";
	innerTSOS = tTrackTR.outermostMeasurementState();
	thePropagator = thePropagatorAlo.product(); 
	//propagationDir = alongToMomentum;
	  
      }
      else 
      {
	TrackRefitterType = "CosmicLike";
	innerTSOS = tTrackTR.innermostMeasurementState();
	thePropagator = thePropagatorOpp.product(); 
	//propagationDir = oppositeToMomentum;
      }	

      
      //Get the transient rechits ------>> ------>> -------->>
      MuonTransientTrackingRecHit::MuonRecHitContainer mySegments =
        theSegmentsAssociator->associate( event, eventSetup, *trackSA, TrackRefitterType );

      //Block For Debugging
      /* 
      edm::LogWarning("Alignment") << "Tag1: " << "StandAlone " << std::endl
                                   << "innerMost " 
                                   << tTrackSA.innermostMeasurementState().globalPosition().perp() << " "
                                   << tTrackSA.innermostMeasurementState().globalPosition().y() << " "
                                   << tTrackSA.innermostMeasurementState().globalDirection().phi() << " "
                                   << "outerMost " 
                                   << tTrackSA.outermostMeasurementState().globalPosition().perp() << " "
                                   << tTrackSA.outermostMeasurementState().globalPosition().y() << " "
                                   << tTrackSA.outermostMeasurementState().globalDirection().phi() << std::endl;
      edm::LogWarning("Alignment") << "Tag1: " << "Tracker " << std::endl
                                   << "innerMost " 
                                   << tTrackTR.innermostMeasurementState().globalPosition().perp() << " "
                                   << tTrackTR.innermostMeasurementState().globalPosition().y() << " "
                                   << tTrackTR.innermostMeasurementState().globalDirection().phi() << " "
                                   << "outerMost " 
                                   << tTrackTR.outermostMeasurementState().globalPosition().perp() << " "
                                   << tTrackTR.outermostMeasurementState().globalPosition().y() << " "
                                   << tTrackTR.outermostMeasurementState().globalDirection().phi() << std::endl;
      edm::LogWarning("Alignment") << "Tag1: " << "Global " << std::endl
                                   << "innerMost " 
                                   << tTrackGR.innermostMeasurementState().globalPosition().perp() << " "
                                   << tTrackGR.innermostMeasurementState().globalPosition().y() << " "
                                   << tTrackGR.innermostMeasurementState().globalDirection().phi() << " "
                                   << "outerMost " 
                                   << tTrackGR.outermostMeasurementState().globalPosition().perp() << " "
                                   << tTrackGR.outermostMeasurementState().globalPosition().y() << " "
                                   << tTrackGR.outermostMeasurementState().globalDirection().phi() << std::endl;

      */
      if( innerTSOS.isValid() )
      {
      
	TrajectoryStateOnSurface innerTSOS_orig = innerTSOS;
      
        //Loop over Associated segments
        for( MuonTransientTrackingRecHit::MuonRecHitContainer::const_iterator rechit = mySegments.begin();
          rechit != mySegments.end(); ++rechit )
        {
          //Get the detector
          const GeomDet* geomDet = (*rechit)->det();

	  if((*rechit)->geographicalId().subdetId() == 1) {
          DTChamberId myChamber( geomDet->geographicalId().rawId() );

	  edm::LogWarning("Alignment") << "TestProp " << " Wheel " << myChamber.wheel() << " Sector " <<  myChamber.sector() << " Chamber " << myChamber.station() << std::endl;
	  }
          //FIXME: Should I really care about this?
          //Otherwise the propagator could throw an exception
	  const Plane* pDest = dynamic_cast<const Plane*>( &geomDet->surface() );
          const Cylinder* cDest = dynamic_cast<const Cylinder*>( &geomDet->surface() );

          if( pDest != 0 || cDest != 0 )
          {   
	    //Propagate
	    //Propagator *updatePropagator=thePropagator->clone();
	    //updatePropagator->setPropagationDirection(propagationDir);
           

            TrajectoryStateOnSurface destiny = thePropagator->propagate( *(innerTSOS.freeState()), geomDet->surface() );
            
            //edm::LogWarning("Alignment") << "TestProp " << " From " << innerTSOS.freeState()->position()
            //                                            << " To "   << geomDet->surface().position()  
            //                                            << " Destiny " << destiny.freeState()->position()  
            //                                            << " RecHit " << geomDet->toGlobal((*rechit)->localPosition())
            //                                            << " Local Destiny " << destiny.localPosition()  
            //                                            << " Local RecHit " << (*rechit)->localPosition() << std::endl;
	    if( !destiny.isValid() || !destiny.hasError() ) continue;
          
            //FIX ME: Use another constructor?
            //Creation of a TrajectoryMeasurement  
	    TrajectoryMeasurement myMeas(destiny, rechit->get());

	    //Block For Debugging
	    //edm::LogWarning("Alignment") << "Tag2: " << "Phi Measured " << std::endl
            //  					 << (*rechit)->localDirection().phi() << " " 
            //					 <<"Phi predicted " 
            //					 << destiny. localDirection().phi() << std::endl;


            //Insert into the Trajectory
            myTraj.push(myMeas);
  
            //new innerTSOS is updated
	    //innerTSOS = destiny;
	  }
        }
        
        trajectoryCollection->push_back(myTraj);
        edm::LogWarning("Alignment") << "Index: " << trajectoryIndex << " " << trackIndex << std::endl;
        trajToTrack_map[trajectoryIndex] = trackIndex;          
        ++trajectoryIndex;
      }  
    }
  }

  edm::OrphanHandle<std::vector<Trajectory> > trajsRef = event.put(trajectoryCollection);
  
  for( trajectoryIndex = 0; trajectoryIndex < tracksTR->size(); ++trajectoryIndex) 
  {      
    edm::Ref<reco::TrackCollection>::key_type trackCounter = trajToTrack_map[trajectoryIndex];
    trajTrackMap->insert(edm::Ref<std::vector<Trajectory> >(trajsRef, trajectoryIndex), edm::Ref<reco::TrackCollection>(tracksTR, trackCounter));
  }
  
  event.put(trajTrackMap);
}

DEFINE_FWK_MODULE(MuonMillepedeTrackRefitter);



