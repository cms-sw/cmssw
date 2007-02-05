// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Conditions database
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
 
// Tracker geom
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

// Alignment
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/Alignment/interface/AlignmentSorter.h"

#include "Alignment/TrackerAlignment/interface/TrackerAlignment.h"

//__________________________________________________________________
//
TrackerAlignment::TrackerAlignment(const edm::EventSetup& setup  ){
	//Create the tracker geometry from the ideal geometry
	// Create the tracker geometry from ideal geometry (first time only)
	edm::ESHandle<DDCompactView> cpv;
	edm::ESHandle<GeometricDet> gD;
	setup.get<IdealGeometryRecord>().get( cpv );
	setup.get<IdealGeometryRecord>().get( gD );
	TrackerGeomBuilderFromGeometricDet trackerBuilder;
	theTracker  = boost::shared_ptr<TrackerGeometry>( trackerBuilder.build(&(*cpv),&(*gD)) );
	
	
	//Retrieve alignable units
	theAlignableTracker = new AlignableTracker(&(*gD),  &(*theTracker));
}
//__________________________________________________________________
//
void TrackerAlignment::moveAlignablePixelEndCaps( int rawid , std::vector<float> local_displacements,  std::vector<float> local_rotations  ){
	
	// Displace and rotate pixelEndCaps
	std::vector<Alignable*> thePixelEndCapsAlignables = theAlignableTracker->pixelEndcapGeomDets();
	for ( std::vector<Alignable*>::iterator iter = thePixelEndCapsAlignables.begin(); iter != thePixelEndCapsAlignables.end(); iter++ ){ 

		// Get the raw ID of the associated GeomDet
		int id = (*iter)->geomDetId().rawId();
		
		// Select the given module
		if ( id == rawid ){
			
			// Convert local to global diplacements
			LocalVector lvector( local_displacements.at(0), local_displacements.at(1), local_displacements.at(2)); 
			GlobalVector gvector = ((*iter)->surface()).toGlobal( lvector );
			
			// global displacement 
			(*iter)->move( gvector );
			
			// local rotation 
			(*iter)->rotateAroundLocalX( local_rotations.at(0) ); // Local X axis rotation
			(*iter)->rotateAroundLocalY( local_rotations.at(1) ); // Local Y axis rotation
			(*iter)->rotateAroundLocalZ( local_rotations.at(2) ); // Local Z axis rotation
			
		}
	}
}
//__________________________________________________________________
//
void TrackerAlignment::moveAlignableEndCaps( int rawid , std::vector<float> local_displacements,  std::vector<float> local_rotations  ){
	
	// Displace and rotate EndCaps
	std::vector<Alignable*> theEndCapsAlignables = theAlignableTracker->endcapGeomDets();
	for ( std::vector<Alignable*>::iterator iter = theEndCapsAlignables.begin(); iter != theEndCapsAlignables.end(); iter++ ){ 
		
		// Get the raw ID of the associated GeomDet
		int id = (*iter)->geomDetId().rawId();
		
		// Select the given module
		if ( id == rawid ){
			
			// Convert local to global diplacements
			LocalVector lvector( local_displacements.at(0), local_displacements.at(1), local_displacements.at(2)); 
			GlobalVector gvector = ((*iter)->surface()).toGlobal( lvector );
			
			// global displacement 
			(*iter)->move( gvector );
			
			// local rotation 
			(*iter)->rotateAroundLocalX( local_rotations.at(0) ); // Local X axis rotation
			(*iter)->rotateAroundLocalY( local_rotations.at(1) ); // Local Y axis rotation
			(*iter)->rotateAroundLocalZ( local_rotations.at(2) ); // Local Z axis rotation
			
		}
	}
}
//__________________________________________________________________
//
void TrackerAlignment::moveAlignablePixelHalfBarrels( int rawid , std::vector<float> local_displacements,  std::vector<float> local_rotations  ){
	
	// Displace and rotate PixelHalfBarrels
	std::vector<Alignable*> thePixelHalfBarrelsAlignables = theAlignableTracker->pixelHalfBarrelGeomDets();
	for ( std::vector<Alignable*>::iterator iter = thePixelHalfBarrelsAlignables.begin(); iter != thePixelHalfBarrelsAlignables.end(); iter++ ){ 
		
		// Get the raw ID of the associated GeomDet
		int id = (*iter)->geomDetId().rawId();
		
		// Select the given module
		if ( id == rawid ){
			
			// Convert local to global diplacements
			LocalVector lvector( local_displacements.at(0), local_displacements.at(1), local_displacements.at(2)); 
			GlobalVector gvector = ((*iter)->surface()).toGlobal( lvector );
			
			// global displacement 
			(*iter)->move( gvector );
			
			// local rotation 
			(*iter)->rotateAroundLocalX( local_rotations.at(0) ); // Local X axis rotation
			(*iter)->rotateAroundLocalY( local_rotations.at(1) ); // Local Y axis rotation
			(*iter)->rotateAroundLocalZ( local_rotations.at(2) ); // Local Z axis rotation
			
		}
	}
}
//__________________________________________________________________
//
void TrackerAlignment::moveAlignableOuterHalfBarrels( int rawid , std::vector<float> local_displacements,  std::vector<float> local_rotations  ){
	
	// Displace and rotate OuterHalfBarrels
	std::vector<Alignable*> theOuterHalfBarrelsAlignables = theAlignableTracker->outerBarrelGeomDets();
	for ( std::vector<Alignable*>::iterator iter = theOuterHalfBarrelsAlignables.begin(); iter != theOuterHalfBarrelsAlignables.end(); iter++ ){ 
		
		// Get the raw ID of the associated GeomDet
		int id = (*iter)->geomDetId().rawId();
		
		// Select the given module
		if ( id == rawid ){
			
			// Convert local to global diplacements
			LocalVector lvector( local_displacements.at(0), local_displacements.at(1), local_displacements.at(2)); 
			GlobalVector gvector = ((*iter)->surface()).toGlobal( lvector );
			
			// global displacement 
			(*iter)->move( gvector );
			
			// local rotation 
			(*iter)->rotateAroundLocalX( local_rotations.at(0) ); // Local X axis rotation
			(*iter)->rotateAroundLocalY( local_rotations.at(1) ); // Local Y axis rotation
			(*iter)->rotateAroundLocalZ( local_rotations.at(2) ); // Local Z axis rotation
			
		}
	}
}
//__________________________________________________________________
//
void TrackerAlignment::moveAlignableInnerHalfBarrels( int rawid , std::vector<float> local_displacements,  std::vector<float> local_rotations  ){
	
	// Displace and rotate InnerHalfBarrels
	std::vector<Alignable*> theInnerHalfBarrelsAlignables = theAlignableTracker->innerBarrelGeomDets();
	for ( std::vector<Alignable*>::iterator iter = theInnerHalfBarrelsAlignables.begin(); iter != theInnerHalfBarrelsAlignables.end(); iter++ ){ 
		
		// Get the raw ID of the associated GeomDet
		int id = (*iter)->geomDetId().rawId();
		
		// Select the given module
		if ( id == rawid ){
			
			// Convert local to global diplacements
			LocalVector lvector( local_displacements.at(0), local_displacements.at(1), local_displacements.at(2)); 
			GlobalVector gvector = ((*iter)->surface()).toGlobal( lvector );
			
			// global displacement 
			(*iter)->move( gvector );
			
			// local rotation 
			(*iter)->rotateAroundLocalX( local_rotations.at(0) ); // Local X axis rotation
			(*iter)->rotateAroundLocalY( local_rotations.at(1) ); // Local Y axis rotation
			(*iter)->rotateAroundLocalZ( local_rotations.at(2) ); // Local Z axis rotation
			
		}
	}
}
//__________________________________________________________________
//
void TrackerAlignment::moveAlignableTIDs( int rawid , std::vector<float> local_displacements,  std::vector<float> local_rotations  ){
	
	// Displace and rotate TIDs
	std::vector<Alignable*> theTIDsAlignables = theAlignableTracker->TIDGeomDets();
	for ( std::vector<Alignable*>::iterator iter = theTIDsAlignables.begin(); iter != theTIDsAlignables.end(); iter++ ){ 
		
		// Get the raw ID of the associated GeomDet
		int id = (*iter)->geomDetId().rawId();
		
		// Select the given module
		if ( id == rawid ){
			
			// Convert local to global diplacements
			LocalVector lvector( local_displacements.at(0), local_displacements.at(1), local_displacements.at(2)); 
			GlobalVector gvector = ((*iter)->surface()).toGlobal( lvector );
			
			// global displacement 
			(*iter)->move( gvector );
			
			// local rotation 
			(*iter)->rotateAroundLocalX( local_rotations.at(0) ); // Local X axis rotation
			(*iter)->rotateAroundLocalY( local_rotations.at(1) ); // Local Y axis rotation
			(*iter)->rotateAroundLocalZ( local_rotations.at(2) ); // Local Z axis rotation
			
		}
	}
}
void TrackerAlignment::saveToDB(void){
	
	// Output POOL-ORA objects
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if( !poolDbService.isAvailable() ) // Die if not available
	throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";

	 // Retrieve, sort and store
  Alignments* alignments = theAlignableTracker->alignments();
	AlignmentErrors* alignmentErrors = theAlignableTracker->alignmentErrors();
	std::sort( alignments->m_align.begin(), alignments->m_align.end(), 
			 lessAlignmentDetId<AlignTransform>() );
  std::sort( alignmentErrors->m_alignError.begin(), alignmentErrors->m_alignError.end(), 
			 lessAlignmentDetId<AlignTransformError>() );

	// Define callback tokens for the payloads
  size_t alignmentsToken = poolDbService->callbackToken("Alignments");
  size_t alignmentErrorsToken = poolDbService->callbackToken("AlignmentErrors");

	poolDbService->newValidityForNewPayload<Alignments>( alignments, 
		poolDbService->endOfTime(),
		alignmentsToken );
  poolDbService->newValidityForNewPayload<AlignmentErrors>( alignmentErrors, 
		poolDbService->endOfTime(),
		alignmentErrorsToken );
}

