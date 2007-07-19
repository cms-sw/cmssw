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
#include "Alignment/MuonAlignment/interface/MuonAlignment.h"

//____________________________________________________________________________________
//
MuonAlignment::MuonAlignment(const edm::EventSetup& setup  ):
  theDTAlignRecordName( "DTAlignmentRcd" ),
  theDTErrorRecordName( "DTAlignmentErrorRcd" ),
  theCSCAlignRecordName( "CSCAlignmentRcd" ),
  theCSCErrorRecordName( "CSCAlignmentErrorRcd" )
{

  // 1. Retrieve geometry from Event setup and create alignable muon
  edm::ESHandle<DTGeometry> dtGeometry;
  edm::ESHandle<CSCGeometry> cscGeometry;

  setup.get<MuonGeometryRecord>().get( dtGeometry );     
  setup.get<MuonGeometryRecord>().get( cscGeometry );

  theAlignableMuon = new AlignableMuon( &(*dtGeometry) , &(*cscGeometry) );

  theAlignableNavigator = new AlignableNavigator( theAlignableMuon );
  
}


//____________________________________________________________________________________
//
void MuonAlignment::moveAlignableLocalCoord( DetId& detid, std::vector<float>& displacements, std::vector<float>& rotations ){

  // Displace and rotate DT an Alignable associated to a GeomDet or GeomDetUnit
  Alignable* theAlignable = theAlignableNavigator->alignableFromDetId( detid );
 
  // Convert local to global diplacements
  LocalVector lvector( displacements.at(0), displacements.at(1), displacements.at(2)); 
  GlobalVector gvector = ( theAlignable->surface()).toGlobal( lvector );

  // global displacement of the chamber
  theAlignable->move( gvector );

  // local rotation of the chamber
  theAlignable->rotateAroundLocalX( rotations.at(0) ); // Local X axis rotation
  theAlignable->rotateAroundLocalY( rotations.at(1) ); // Local Y axis rotation
  theAlignable->rotateAroundLocalZ( rotations.at(2) ); // Local Z axis rotation

}

//____________________________________________________________________________________
//
void MuonAlignment::moveAlignableGlobalCoord( DetId& detid, std::vector<float>& displacements, std::vector<float>& rotations ){

  // Displace and rotate DT an Alignable associated to a GeomDet or GeomDetUnit
  Alignable* theAlignable = theAlignableNavigator->alignableFromDetId( detid );
 
  // Convert std::vector to GlobalVector
  GlobalVector gvector( displacements.at(0), displacements.at(1), displacements.at(2)); 

  // global displacement of the chamber
  theAlignable->move( gvector );

  // local rotation of the chamber
  theAlignable->rotateAroundGlobalX( rotations.at(0) ); // Global X axis rotation
  theAlignable->rotateAroundGlobalY( rotations.at(1) ); // Global Y axis rotation
  theAlignable->rotateAroundGlobalZ( rotations.at(2) ); // Global Z axis rotation

}

//____________________________________________________________________________________
// Code needed to store alignments to DB

void MuonAlignment::saveDTtoDB(void) {
   // Call service
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if( !poolDbService.isAvailable() ) // Die if not available
	throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";

  // Get alignments and errors
  Alignments*      dt_Alignments       = theAlignableMuon->dtAlignments() ;
  AlignmentErrors* dt_AlignmentErrors  = theAlignableMuon->dtAlignmentErrors();

  // Store DT alignments and errors
  if ( poolDbService->isNewTagRequest(theDTAlignRecordName) ){
   poolDbService->createNewIOV<Alignments>( &(*dt_Alignments), 
	                                    poolDbService->endOfTime(), 
                                            theDTAlignRecordName );
  } else {
    poolDbService->appendSinceTime<Alignments>( &(*dt_Alignments),
                                                poolDbService->currentTime(), 
                                                theDTAlignRecordName );
  }
      
  if ( poolDbService->isNewTagRequest(theDTErrorRecordName) ){
   poolDbService->createNewIOV<AlignmentErrors>( &(*dt_AlignmentErrors),
                                                 poolDbService->endOfTime(), 
                                                 theDTErrorRecordName );
  } else {
   poolDbService->appendSinceTime<AlignmentErrors>( &(*dt_AlignmentErrors),
                                                    poolDbService->currentTime(),
                                                    theDTErrorRecordName );
  }							  
}

void MuonAlignment::saveCSCtoDB(void) {
   // Call service
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if( !poolDbService.isAvailable() ) // Die if not available
	throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";

  // Get alignments and errors
  Alignments*      csc_Alignments      = theAlignableMuon->cscAlignments();
  AlignmentErrors* csc_AlignmentErrors = theAlignableMuon->cscAlignmentErrors();

  // Store CSC alignments and errors
  if ( poolDbService->isNewTagRequest(theCSCAlignRecordName) ){
   poolDbService->createNewIOV<Alignments>( &(*csc_Alignments), 
	                                    poolDbService->endOfTime(), 
                                            theCSCAlignRecordName );
  } else {
    poolDbService->appendSinceTime<Alignments>( &(*csc_Alignments),
                                                poolDbService->currentTime(), 
                                                theCSCAlignRecordName );
  }
      
  if ( poolDbService->isNewTagRequest(theCSCErrorRecordName) ){
   poolDbService->createNewIOV<AlignmentErrors>( &(*csc_AlignmentErrors),
                                                 poolDbService->endOfTime(), 
                                                 theCSCErrorRecordName );
  } else {
   poolDbService->appendSinceTime<AlignmentErrors>( &(*csc_AlignmentErrors),
                                                    poolDbService->currentTime(),
                                                    theCSCErrorRecordName );
  }							  
}

void MuonAlignment::saveToDB(void) {
   saveDTtoDB();
   saveCSCtoDB();
}
