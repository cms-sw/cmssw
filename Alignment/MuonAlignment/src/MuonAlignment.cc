//
//
//
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Conditions database
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

// Muon geom
#include "Geometry/Records/interface/MuonGeometryRecord.h"

// Alignment
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/Alignment/interface/AlignmentSorter.h"
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "Alignment/MuonAlignment/interface/MuonAlignment.h"

//____________________________________________________________________________________
//
MuonAlignment::MuonAlignment(const edm::EventSetup& setup ){

  // 1. Retrieve geometry from Event setup and create alignable muon
  edm::ESHandle<DTGeometry> dtGeometry;
  edm::ESHandle<CSCGeometry> cscGeometry;

  setup.get<MuonGeometryRecord>().get( dtGeometry );     
  setup.get<MuonGeometryRecord>().get( cscGeometry );

  theAlignableMuon = new AlignableMuon( &(*dtGeometry) , &(*cscGeometry) );
 
}


//____________________________________________________________________________________
//
void MuonAlignment::moveDTChamber( int rawid , std::vector<float> local_displacements,  std::vector<float> local_rotations  ){

  // Displace and rotate DT chambers
  std::vector<Alignable*> theDTAlignables = theAlignableMuon->DTChambers();
  for ( std::vector<Alignable*>::iterator iter = theDTAlignables.begin();
        iter != theDTAlignables.end(); iter++ ){ 

    // Get the raw ID of the associated GeomDet
    int id = (*iter)->geomDetId().rawId();

    // Select the given chamber
    if ( id == rawid ){
          

      // Convert local to global diplacements
      LocalVector lvector( local_displacements.at(0), local_displacements.at(1), local_displacements.at(2)); 
      GlobalVector gvector = ((*iter)->surface()).toGlobal( lvector );

      // global displacement of the chamber
      (*iter)->move( gvector );
      
      // local rotation of the chamber
      (*iter)->rotateAroundLocalX( local_rotations.at(0) ); // Local X axis rotation
      (*iter)->rotateAroundLocalY( local_rotations.at(1) ); // Local Y axis rotation
      (*iter)->rotateAroundLocalZ( local_rotations.at(2) ); // Local Z axis rotation

    }
   
  }

}
//____________________________________________________________________________________
//
void MuonAlignment::moveCSCChamber( int rawid , std::vector<float> local_displacements, std::vector<float> local_rotations  ){

  // Displace and rotate CSC chambers
  std::vector<Alignable*> theCSCAlignables = theAlignableMuon->CSCChambers();
  for ( std::vector<Alignable*>::iterator iter = theCSCAlignables.begin();
	iter != theCSCAlignables.end(); iter++ ){ 

    // Get the raw ID of the associated GeomDet
    int id = (*iter)->geomDetId().rawId();

    // Select the given chamber
    if ( id == rawid ){

      // Convert local to global diplacements
      LocalVector lvector( local_displacements.at(0), local_displacements.at(1), local_displacements.at(2));
      GlobalVector gvector = ((*iter)->surface()).toGlobal( lvector );

      // global diplacement of the chamber
      (*iter)->move( gvector );

      // local rotation of the chamber
      (*iter)->rotateAroundLocalX( local_rotations.at(0) ); // Local X axis rotation
      (*iter)->rotateAroundLocalY( local_rotations.at(1) ); // Local Y axis rotation
      (*iter)->rotateAroundLocalZ( local_rotations.at(2) ); // Local Z axis rotation

    }

  }  

}


//____________________________________________________________________________________
// Code needed to store alignments to DB
void MuonAlignment::saveToDB( void ){
   
   // Call service
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if( !poolDbService.isAvailable() ) // Die if not available
	throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";

  // Retrieve muon barrel alignments and errors
  Alignments*      dtAlignments       = theAlignableMuon->DTBarrel().front()->alignments();
  AlignmentErrors* dtAlignmentErrors = theAlignableMuon->DTBarrel().front()->alignmentErrors();

  // Retrieve muon endcaps alignments and errors (there are two endcaps...)
  Alignments* cscEndCap1    = theAlignableMuon->CSCEndcaps().front()->alignments();
  Alignments* cscEndCap2    = theAlignableMuon->CSCEndcaps().back()->alignments();
  Alignments* cscAlignments = new Alignments();
  std::copy( cscEndCap1->m_align.begin(), cscEndCap1->m_align.end(), back_inserter( cscAlignments->m_align ) );
  std::copy( cscEndCap2->m_align.begin(), cscEndCap2->m_align.end(), back_inserter( cscAlignments->m_align ) );

  AlignmentErrors* cscEndCap1Errors = theAlignableMuon->CSCEndcaps().front()->alignmentErrors();
  AlignmentErrors* cscEndCap2Errors = theAlignableMuon->CSCEndcaps().back()->alignmentErrors();
  AlignmentErrors* cscAlignmentErrors    = new AlignmentErrors();
  std::copy(cscEndCap1Errors->m_alignError.begin(), cscEndCap1Errors->m_alignError.end(), 
             back_inserter(cscAlignmentErrors->m_alignError) );
  std::copy(cscEndCap2Errors->m_alignError.begin(), cscEndCap2Errors->m_alignError.end(),
             back_inserter(cscAlignmentErrors->m_alignError) );

  // Sort by DetID
  std::sort( dtAlignments->m_align.begin(),  dtAlignments->m_align.end(),  lessAlignmentDetId<AlignTransform>() );
  std::sort( dtAlignmentErrors->m_alignError.begin(),  dtAlignmentErrors->m_alignError.end(),  lessAlignmentDetId<AlignTransformError>() );

  std::sort( cscAlignments->m_align.begin(), cscAlignments->m_align.end(), lessAlignmentDetId<AlignTransform>() );
  std::sort( cscAlignmentErrors->m_alignError.begin(), cscAlignmentErrors->m_alignError.end(), lessAlignmentDetId<AlignTransformError>() );

  // Define callback tokens for the records 
  size_t dtAlignmentsToken = poolDbService->callbackToken("dtAlignments");
  size_t dtAlignmentErrorsToken  = poolDbService->callbackToken("dtAlignmentErrors");

  size_t cscAlignmentsToken = poolDbService->callbackToken("cscAlignments");
  size_t cscAlignmentErrorsToken = poolDbService->callbackToken("cscAlignmentErrors");

  // Store in the database
  poolDbService->newValidityForNewPayload<Alignments>( dtAlignments, poolDbService->endOfTime(), dtAlignmentsToken );
  poolDbService->newValidityForNewPayload<AlignmentErrors>( dtAlignmentErrors, poolDbService->endOfTime(), dtAlignmentErrorsToken );

  poolDbService->newValidityForNewPayload<Alignments>( cscAlignments, poolDbService->endOfTime(), cscAlignmentsToken );
  poolDbService->newValidityForNewPayload<AlignmentErrors>( cscAlignmentErrors, poolDbService->endOfTime(), cscAlignmentErrorsToken );

}

