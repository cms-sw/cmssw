// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

// Conditions database
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

// Muon geom
#include "Geometry/Records/interface/MuonGeometryRecord.h"

// Alignment
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "CondFormats/Alignment/interface/SurveyErrors.h"
#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "Alignment/MuonAlignment/interface/MuonAlignment.h"

//____________________________________________________________________________________
//
MuonAlignment::MuonAlignment(const edm::EventSetup& setup  ):
  theDTAlignRecordName( "DTAlignmentRcd" ),
  theDTErrorRecordName( "DTAlignmentErrorRcd" ),
  theCSCAlignRecordName( "CSCAlignmentRcd" ),
  theCSCErrorRecordName( "CSCAlignmentErrorRcd" ),
  theDTSurveyRecordName( "DTSurveyRcd" ),
  theDTSurveyErrorRecordName( "DTSurveyErrorRcd" ),
  theCSCSurveyRecordName( "CSCSurveyRcd" ),
  theCSCSurveyErrorRecordName( "CSCSurveyErrorRcd" )
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

void MuonAlignment::recursiveList(std::vector<Alignable*> alignables, std::vector<Alignable*> &theList) {
   for (std::vector<Alignable*>::const_iterator alignable = alignables.begin();  alignable != alignables.end();  ++alignable) {
      recursiveList((*alignable)->components(), theList);
      theList.push_back(*alignable);
   }
}

void MuonAlignment::saveDTSurveyToDB(void) {
   // Call service
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if( !poolDbService.isAvailable() ) // Die if not available
     throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";

  // Get alignments and errors
  Alignments *dtAlignments = new Alignments();
  SurveyErrors *dtSurveyErrors = new SurveyErrors();

  std::vector<Alignable*> alignableList;
  recursiveList(theAlignableMuon->DTBarrel(), alignableList);

  for (std::vector<Alignable*>::const_iterator alignable = alignableList.begin();  alignable != alignableList.end();  ++alignable) {
     Alignable *aliid = *alignable;
     while (aliid->id() == 0) aliid = aliid->components()[0];

     const align::PositionType &pos = (*alignable)->survey()->position();
     const align::RotationType &rot = (*alignable)->survey()->rotation();

     AlignTransform value(CLHEP::Hep3Vector(pos.x(), pos.y(), pos.z()),
			    CLHEP::HepRotation(CLHEP::HepRep3x3(rot.xx(), rot.xy(), rot.xz(),
								      rot.yx(), rot.yy(), rot.yz(),
								      rot.zx(), rot.zy(), rot.zz())),
			    aliid->id());
     SurveyError error((*alignable)->alignableObjectId(), aliid->id(), (*alignable)->survey()->errors());
     
     dtAlignments->m_align.push_back(value);
     dtSurveyErrors->m_surveyErrors.push_back(error);
  }

  // Store DT alignments and errors
  poolDbService->writeOne<Alignments>( &(*dtAlignments), poolDbService->currentTime(), theDTSurveyRecordName);
  poolDbService->writeOne<SurveyErrors>( &(*dtSurveyErrors), poolDbService->currentTime(), theDTSurveyErrorRecordName);
}

void MuonAlignment::saveCSCSurveyToDB(void) {
   // Call service
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if( !poolDbService.isAvailable() ) // Die if not available
     throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";

  // Get alignments and errors
  Alignments *cscAlignments = new Alignments();
  SurveyErrors *cscSurveyErrors = new SurveyErrors();

  std::vector<Alignable*> alignableList;
  recursiveList(theAlignableMuon->CSCEndcaps(), alignableList);

  for (std::vector<Alignable*>::const_iterator alignable = alignableList.begin();  alignable != alignableList.end();  ++alignable) {
     Alignable *aliid = *alignable;
     while (aliid->id() == 0) aliid = aliid->components()[0];

     const align::PositionType &pos = (*alignable)->survey()->position();
     const align::RotationType &rot = (*alignable)->survey()->rotation();

     AlignTransform value(CLHEP::Hep3Vector(pos.x(), pos.y(), pos.z()),
			    CLHEP::HepRotation(CLHEP::HepRep3x3(rot.xx(), rot.xy(), rot.xz(),
								      rot.yx(), rot.yy(), rot.yz(),
								      rot.zx(), rot.zy(), rot.zz())),
			    aliid->id());
     SurveyError error((*alignable)->alignableObjectId(), aliid->id(), (*alignable)->survey()->errors());
     
     cscAlignments->m_align.push_back(value);
     cscSurveyErrors->m_surveyErrors.push_back(error);
  }

  // Store CSC alignments and errors
  poolDbService->writeOne<Alignments>( &(*cscAlignments), poolDbService->currentTime(), theCSCSurveyRecordName);
  poolDbService->writeOne<SurveyErrors>( &(*cscSurveyErrors), poolDbService->currentTime(), theCSCSurveyErrorRecordName);
}

void MuonAlignment::saveSurveyToDB(void) {
   saveDTSurveyToDB();
   saveCSCSurveyToDB();
}

void MuonAlignment::saveDTtoDB(void) {
   // Call service
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if( !poolDbService.isAvailable() ) // Die if not available
	throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";

  // Get alignments and errors
  Alignments*      dt_Alignments       = theAlignableMuon->dtAlignments() ;
  AlignmentErrors* dt_AlignmentErrors  = theAlignableMuon->dtAlignmentErrors();

  // Store DT alignments and errors
  poolDbService->writeOne<Alignments>( &(*dt_Alignments), poolDbService->currentTime(), theDTAlignRecordName);
  poolDbService->writeOne<AlignmentErrors>( &(*dt_AlignmentErrors), poolDbService->currentTime(), theDTErrorRecordName);
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
  poolDbService->writeOne<Alignments>( &(*csc_Alignments), poolDbService->currentTime(), theCSCAlignRecordName);
  poolDbService->writeOne<AlignmentErrors>( &(*csc_AlignmentErrors), poolDbService->currentTime(), theCSCErrorRecordName);
}

void MuonAlignment::saveToDB(void) {
   saveDTtoDB();
   saveCSCtoDB();
}
