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

#include "Alignment/MuonAlignment/interface/MuonAlignmentOutputXML.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"

//____________________________________________________________________________________
//
void MuonAlignment::init()
{
   theDTAlignRecordName = "DTAlignmentRcd";
   theDTErrorRecordName = "DTAlignmentErrorRcd";
   theCSCAlignRecordName = "CSCAlignmentRcd";
   theCSCErrorRecordName = "CSCAlignmentErrorRcd";
   theDTSurveyRecordName = "DTSurveyRcd";
   theDTSurveyErrorRecordName = "DTSurveyErrorRcd";
   theCSCSurveyRecordName = "CSCSurveyRcd";
   theCSCSurveyErrorRecordName = "CSCSurveyErrorRcd";
   theAlignableMuon = NULL;
   theAlignableNavigator = NULL;
}

MuonAlignment::MuonAlignment( const edm::EventSetup& iSetup )
{
   init();

   edm::ESHandle<DTGeometry> dtGeometry;
   edm::ESHandle<CSCGeometry> cscGeometry;
   iSetup.get<MuonGeometryRecord>().get( dtGeometry );     
   iSetup.get<MuonGeometryRecord>().get( cscGeometry );

   theAlignableMuon = new AlignableMuon( &(*dtGeometry) , &(*cscGeometry) );
   theAlignableNavigator = new AlignableNavigator( theAlignableMuon );
}

MuonAlignment::MuonAlignment( const edm::EventSetup& iSetup, const MuonAlignmentInputMethod& input )
{
   init();

   theAlignableMuon = input.newAlignableMuon( iSetup );
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
//
void MuonAlignment::recursiveList(std::vector<Alignable*> alignables, std::vector<Alignable*> &theList) {
   for (std::vector<Alignable*>::const_iterator alignable = alignables.begin();  alignable != alignables.end();  ++alignable) {
      recursiveList((*alignable)->components(), theList);
      theList.push_back(*alignable);
   }
}

//____________________________________________________________________________________
//
void MuonAlignment::recursiveMap(std::vector<Alignable*> alignables, std::map<align::ID, Alignable*> &theMap) {
   for (std::vector<Alignable*>::const_iterator alignable = alignables.begin();  alignable != alignables.end();  ++alignable) {
      unsigned int rawId = (*alignable)->geomDetId().rawId();
      if (rawId != 0) {
	 theMap[rawId] = *alignable;
      }
      recursiveMap((*alignable)->components(), theMap);
   }
}

//____________________________________________________________________________________
//
void MuonAlignment::recursiveStructureMap(std::vector<Alignable*> alignables, std::map<std::pair<align::StructureType, align::ID>, Alignable*> &theMap) {
   for (std::vector<Alignable*>::const_iterator alignable = alignables.begin();  alignable != alignables.end();  ++alignable) {
      theMap[std::pair<align::StructureType, align::ID>((*alignable)->alignableObjectId(), (*alignable)->id())] = *alignable;
      recursiveStructureMap((*alignable)->components(), theMap);
   }
}

//____________________________________________________________________________________
//
void MuonAlignment::copyAlignmentToSurvey(double shiftErr, double angleErr) {
   std::map<align::ID, Alignable*> alignableMap;
   recursiveMap(theAlignableMuon->DTBarrel(), alignableMap);
   recursiveMap(theAlignableMuon->CSCEndcaps(), alignableMap);

   // Set the survey error to the alignable error, expanding the matrix as needed
   AlignmentErrors* dtAlignmentErrors = theAlignableMuon->dtAlignmentErrors();
   AlignmentErrors* cscAlignmentErrors = theAlignableMuon->cscAlignmentErrors();
   std::vector<AlignTransformError> alignmentErrors;
   std::copy(dtAlignmentErrors->m_alignError.begin(), dtAlignmentErrors->m_alignError.end(), std::back_inserter(alignmentErrors));
   std::copy(cscAlignmentErrors->m_alignError.begin(), cscAlignmentErrors->m_alignError.end(), std::back_inserter(alignmentErrors));

   for (std::vector<AlignTransformError>::const_iterator alignmentError = alignmentErrors.begin();
	alignmentError != alignmentErrors.end();
	++alignmentError) {
      align::ErrorMatrix matrix6x6 = ROOT::Math::SMatrixIdentity();
      CLHEP::HepSymMatrix matrix3x3 = alignmentError->matrix();

      for (int i = 0;  i < 3;  i++) {
	 for (int j = 0;  j < 3;  j++) {
	    matrix6x6(i, j) = matrix3x3(i+1, j+1);
	 }
      }
      matrix6x6(3,3) = angleErr;
      matrix6x6(4,4) = angleErr;
      matrix6x6(5,5) = angleErr;

      Alignable *alignable = alignableMap[alignmentError->rawId()];
      alignable->setSurvey(new SurveyDet(alignable->surface(), matrix6x6));
   }

   fillGapsInSurvey(shiftErr, angleErr);
}

//____________________________________________________________________________________
//

void MuonAlignment::fillGapsInSurvey(double shiftErr, double angleErr) {
   // get all the ones we missed
   std::map<std::pair<align::StructureType, align::ID>, Alignable*> alignableStructureMap;
   recursiveStructureMap(theAlignableMuon->DTBarrel(), alignableStructureMap);
   recursiveStructureMap(theAlignableMuon->CSCEndcaps(), alignableStructureMap);

   for (std::map<std::pair<align::StructureType, align::ID>, Alignable*>::const_iterator iter = alignableStructureMap.begin();
	iter != alignableStructureMap.end();
	++iter) {
      if (iter->second->survey() == NULL) {
	 align::ErrorMatrix matrix6x6 = ROOT::Math::SMatrixIdentity();
	 matrix6x6(0,0) = shiftErr;
	 matrix6x6(1,1) = shiftErr;
	 matrix6x6(2,2) = shiftErr;
	 matrix6x6(3,3) = angleErr;
	 matrix6x6(4,4) = angleErr;
	 matrix6x6(5,5) = angleErr;
	 iter->second->setSurvey(new SurveyDet(iter->second->surface(), matrix6x6));
      }
   }
}

//____________________________________________________________________________________
//
void MuonAlignment::recursiveCopySurveyToAlignment(Alignable *alignable) {
   if (alignable->survey() != NULL) {
      const SurveyDet *survey = alignable->survey();

      align::PositionType pos = survey->position();
      align::RotationType rot = survey->rotation();

      align::PositionType oldpos = alignable->globalPosition();
      align::RotationType oldrot = alignable->globalRotation();
      alignable->move(GlobalVector(-oldpos.x(), -oldpos.y(), -oldpos.z()));
      alignable->rotateInGlobalFrame(oldrot.transposed());
      alignable->rotateInGlobalFrame(rot);
      alignable->move(GlobalVector(pos.x(), pos.y(), pos.z()));

      align::ErrorMatrix matrix6x6 = survey->errors();  // start from 0,0
      CLHEP::HepSymMatrix matrix3x3(3);                 // start from 1,1
      for (int i = 0;  i < 3;  i++) {
	 for (int j = 0;  j < 3;  j++) {
	    matrix3x3(i+1, j+1) = matrix6x6(i, j);
	 }
      }

      // this sets APEs at this level and (since 2nd argument is true) all lower levels
      alignable->setAlignmentPositionError(AlignmentPositionError(GlobalError(matrix3x3)), true);
   }

   // do lower levels afterward to thwart the cumulative setting of APEs
   std::vector<Alignable*> components = alignable->components();
   for (std::vector<Alignable*>::const_iterator comp = components.begin();  comp != components.end();  ++comp) {
      recursiveCopySurveyToAlignment(*comp);
   }
}

void MuonAlignment::copySurveyToAlignment() {
   recursiveCopySurveyToAlignment(theAlignableMuon);
}

//____________________________________________________________________________________
// Code needed to store alignments to DB

void MuonAlignment::writeXML(const edm::ParameterSet iConfig, const edm::EventSetup &iSetup) {
   MuonAlignmentOutputXML(iConfig).write(theAlignableMuon, iSetup);
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
     const align::PositionType &pos = (*alignable)->survey()->position();
     const align::RotationType &rot = (*alignable)->survey()->rotation();

     AlignTransform value(CLHEP::Hep3Vector(pos.x(), pos.y(), pos.z()),
			    CLHEP::HepRotation(CLHEP::HepRep3x3(rot.xx(), rot.xy(), rot.xz(),
								      rot.yx(), rot.yy(), rot.yz(),
								      rot.zx(), rot.zy(), rot.zz())),
			    (*alignable)->id());
     SurveyError error((*alignable)->alignableObjectId(), (*alignable)->id(), (*alignable)->survey()->errors());
     
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
     const align::PositionType &pos = (*alignable)->survey()->position();
     const align::RotationType &rot = (*alignable)->survey()->rotation();

     AlignTransform value(CLHEP::Hep3Vector(pos.x(), pos.y(), pos.z()),
			    CLHEP::HepRotation(CLHEP::HepRep3x3(rot.xx(), rot.xy(), rot.xz(),
								      rot.yx(), rot.yy(), rot.yz(),
								      rot.zx(), rot.zy(), rot.zz())),
			    (*alignable)->id());
     SurveyError error((*alignable)->alignableObjectId(), (*alignable)->id(), (*alignable)->survey()->errors());
     
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
