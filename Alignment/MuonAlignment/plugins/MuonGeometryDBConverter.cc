// -*- C++ -*-
//
// Package:    MuonGeometryDBConverter
// Class:      MuonGeometryDBConverter
// 
/**\class MuonGeometryDBConverter MuonGeometryDBConverter.cc Alignment/MuonAlignment/plugins/MuonGeometryDBConverter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Pivarski
//         Created:  Sat Feb 16 00:04:55 CST 2008
// $Id$
//
//


// system include files
#include <memory>
#include <algorithm>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/MuonAlignment/interface/MuonScenarioBuilder.h"
#include "Alignment/CommonAlignment/interface/Alignable.h" 
#include "Alignment/CommonAlignment/interface/SurveyDet.h" 
#include "CondFormats/Alignment/interface/AlignmentErrors.h" 
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h"
#include "Geometry/CSCGeometryBuilder/src/CSCGeometryBuilderFromDDD.h"
#include "Geometry/TrackingGeometryAligner/interface/GeometryAligner.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentErrorRcd.h"
#include "CondFormats/Alignment/interface/SurveyErrors.h"
#include "CondFormats/AlignmentRecord/interface/DTSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTSurveyErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCSurveyErrorRcd.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/CommonAlignment/interface/StructureType.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "CondFormats/Alignment/interface/Definitions.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CLHEP/Matrix/SymMatrix.h"

//
// class decleration
//

class MuonGeometryDBConverter : public edm::EDAnalyzer {
   public:
      explicit MuonGeometryDBConverter(const edm::ParameterSet&);
      ~MuonGeometryDBConverter();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      void recursiveList(std::vector<Alignable*> alignables, std::vector<Alignable*> &theList);
      void recursiveMap(std::vector<Alignable*> alignables, std::map<align::ID, Alignable*> &theMap, bool allowDetUnit);
      void recursiveStructureMap(std::vector<Alignable*> alignables, std::map<std::pair<align::StructureType, align::ID>, Alignable*> &theMap);
      void recursiveReadIn(std::vector<Alignable*> alignables,
			   align::PositionType globalPosition, align::RotationType globalRotation,
			   edm::ParameterSet pset, bool DT, bool survey);
      void recursivePrintOut(std::vector<Alignable*> alignables,
			     align::PositionType globalPosition, align::RotationType globalRotation,
			     std::ofstream &output, int depth, bool DT, bool survey);

      // ----------member data ---------------------------
      double m_missingErrorTranslation, m_missingErrorAngle;
      bool m_alwaysEulerAngles;

      std::string m_CSCInputMode;
      bool m_CSCInputSurvey;
      std::string m_CSCAlignmentsLabel, m_CSCErrorsLabel;
      edm::ParameterSet m_CSCInput;

      std::string m_CSCOutputMode;
      bool m_CSCOutputSurvey;
      std::string m_CSCOutputFileName;
      std::string m_CSCOutputBlockName;

      AlignableMuon *m_alignableMuon;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MuonGeometryDBConverter::MuonGeometryDBConverter(const edm::ParameterSet &iConfig) {
   m_missingErrorTranslation = iConfig.getUntrackedParameter("missingErrorTranslation", 10.);
   m_missingErrorAngle = iConfig.getUntrackedParameter("missingErrorAngle", 10.);
   m_alwaysEulerAngles = iConfig.getUntrackedParameter("alwaysEulerAngles", false);

   m_CSCInputMode = iConfig.getParameter<std::string>("CSCInputMode");
   m_CSCInputSurvey = iConfig.getParameter<bool>("CSCInputSurvey");
   if (m_CSCInputMode == std::string("ideal")) {
      // no extra parameters needed
   }
   else if (m_CSCInputMode == std::string("db")) {
      m_CSCAlignmentsLabel = iConfig.getUntrackedParameter<std::string>("CSCAlignmentsLabel", "");
      m_CSCErrorsLabel = iConfig.getUntrackedParameter<std::string>("CSCErrorsLabel", "");
   }
   else if (m_CSCInputMode == std::string("cff")) {
      m_CSCInput = iConfig.getParameter<edm::ParameterSet>("CSCInput");
   }
   else {
      throw cms::Exception("BadConfig") << "CSCInputMode must be one of \"ideal\", \"db\", \"cfg\"." << std::endl;
   }

   m_CSCOutputMode = iConfig.getParameter<std::string>("CSCOutputMode");
   m_CSCOutputSurvey = iConfig.getParameter<bool>("CSCOutputSurvey");
   if (m_CSCOutputMode == std::string("none")) {
      // no extra parameters needed
   }
   else if (m_CSCOutputMode == std::string("db")) {
      // no extra parameters needed
   }
   else if (m_CSCOutputMode == std::string("cff")) {
      m_CSCOutputFileName = iConfig.getParameter<std::string>("CSCOutputFileName");
      m_CSCOutputBlockName = iConfig.getParameter<std::string>("CSCOutputBlockName");
   }
   else {
      throw cms::Exception("BadConfig") << "CSCOutputMode must be one of \"none\", \"db\", \"cfg\"." << std::endl;
   }
}

MuonGeometryDBConverter::~MuonGeometryDBConverter() { } 

// ------------ method called to for each event  ------------
void
MuonGeometryDBConverter::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
   throw cms::Exception("BadConfig") << "Set maxEvents.input to 0.  (Your output is okay.)" << std::endl;
}

void MuonGeometryDBConverter::recursiveList(std::vector<Alignable*> alignables, std::vector<Alignable*> &theList) {
   for (std::vector<Alignable*>::const_iterator alignable = alignables.begin();  alignable != alignables.end();  ++alignable) {
      theList.push_back(*alignable);
      recursiveList((*alignable)->components(), theList);
   }
}

void MuonGeometryDBConverter::recursiveMap(std::vector<Alignable*> alignables, std::map<align::ID, Alignable*> &theMap, bool allowDetUnit) {
   for (std::vector<Alignable*>::const_iterator alignable = alignables.begin();  alignable != alignables.end();  ++alignable) {
      if ((*alignable)->id() != 0) {
	 if (allowDetUnit  ||  (*alignable)->alignableObjectId() != align::AlignableDetUnit) {
	    theMap[(*alignable)->id()] = *alignable;
	 }
      }
      recursiveMap((*alignable)->components(), theMap, allowDetUnit);
   }
}

void MuonGeometryDBConverter::recursiveStructureMap(std::vector<Alignable*> alignables, std::map<std::pair<align::StructureType, align::ID>, Alignable*> &theStructureMap) {
   for (std::vector<Alignable*>::const_iterator alignable = alignables.begin();  alignable != alignables.end();  ++alignable) {
      Alignable *aliid = *alignable;
      while (aliid->id() == 0) aliid = aliid->components()[0];
      theStructureMap[std::pair<align::StructureType, align::ID>((*alignable)->alignableObjectId(), aliid->id())] = *alignable;
      recursiveStructureMap((*alignable)->components(), theStructureMap);
   }
}

bool sortOrder_AlignTransform(AlignTransform a, AlignTransform b) { return a.rawId() < b.rawId(); }
bool sortOrder_AlignTransformError(AlignTransformError a, AlignTransformError b) { return a.rawId() < b.rawId(); }

// ------------ method called once each job just before starting event loop  ------------
void 
MuonGeometryDBConverter::beginJob(const edm::EventSetup &iSetup) {
   edm::ESHandle<DDCompactView> cpv;
   iSetup.get<IdealGeometryRecord>().get(cpv);

   edm::ESHandle<MuonDDDConstants> mdc;
   iSetup.get<MuonNumberingRecord>().get(mdc);
   DTGeometryBuilderFromDDD DTGeometryBuilder;
   CSCGeometryBuilderFromDDD CSCGeometryBuilder;
 
   DTGeometry *dtGeometry = DTGeometryBuilder.build(&(*cpv), *mdc);
   boost::shared_ptr<CSCGeometry> boost_cscGeometry(new CSCGeometry);
   CSCGeometryBuilder.build(boost_cscGeometry, &(*cpv), *mdc);
   CSCGeometry *cscGeometry = &(*boost_cscGeometry);
 
   GeometryAligner aligner;

   if (m_CSCInputMode == std::string("ideal")) {
      m_alignableMuon = new AlignableMuon(dtGeometry, cscGeometry);
      std::vector<Alignable*> alignableList;
      recursiveList(m_alignableMuon->CSCEndcaps(), alignableList);

      for (std::vector<Alignable*>::const_iterator alignable = alignableList.begin();  alignable != alignableList.end();  ++alignable) {
	 (*alignable)->setAlignmentPositionError(AlignmentPositionError(m_missingErrorTranslation, m_missingErrorTranslation, m_missingErrorTranslation));

	 align::ErrorMatrix matrix6by6 = ROOT::Math::SMatrixIdentity();
	 matrix6by6(0,0) = m_missingErrorTranslation;
	 matrix6by6(1,1) = m_missingErrorTranslation;
	 matrix6by6(2,2) = m_missingErrorTranslation;
	 matrix6by6(3,3) = m_missingErrorAngle;
	 matrix6by6(4,4) = m_missingErrorAngle;
	 matrix6by6(5,5) = m_missingErrorAngle;
	 (*alignable)->setSurvey(new SurveyDet((*alignable)->surface(), matrix6by6));
      }
   }
   else if (m_CSCInputMode == std::string("db")) {
      if (m_CSCInputSurvey) {
	 edm::ESHandle<Alignments> cscSurvey;
	 edm::ESHandle<SurveyErrors> cscSurveyErrors;
	 iSetup.get<CSCSurveyRcd>().get(m_CSCAlignmentsLabel, cscSurvey);
	 iSetup.get<CSCSurveyErrorRcd>().get(m_CSCErrorsLabel, cscSurveyErrors);

	 Alignments cscAlignments;
	 AlignmentErrors cscAlignmentErrors;
	 std::vector<AlignTransform>::const_iterator alignment = cscSurvey->m_align.begin();
	 std::vector<SurveyError>::const_iterator surveyError = cscSurveyErrors->m_surveyErrors.begin();

	 for (;  alignment != cscSurvey->m_align.end()  &&  surveyError != cscSurveyErrors->m_surveyErrors.end();  ++alignment, ++surveyError) {
	    if (surveyError->structureType() == align::AlignableCSCChamber  ||  surveyError->structureType() == align::AlignableDet) {
	       align::ErrorMatrix matrix6by6 = surveyError->matrix();  // start from 0,0
	       CLHEP::HepSymMatrix matrix3by3(3);                      // start from 1,1
	       for (int i = 0;  i < 3;  i++) {
		  for (int j = 0;  j < 3;  j++) {
		     matrix3by3(i+1, j+1) = matrix6by6(i, j);
		  }
	       }
	       cscAlignments.m_align.push_back(AlignTransform(*alignment));
	       cscAlignmentErrors.m_alignError.push_back(AlignTransformError(matrix3by3, surveyError->rawId()));
	    }
	 }

	 std::sort(cscAlignments.m_align.begin(), cscAlignments.m_align.end(), sortOrder_AlignTransform);
	 std::sort(cscAlignmentErrors.m_alignError.begin(), cscAlignmentErrors.m_alignError.end(), sortOrder_AlignTransformError);

	 aligner.applyAlignments<CSCGeometry>(cscGeometry, &cscAlignments, &cscAlignmentErrors);

	 m_alignableMuon = new AlignableMuon(dtGeometry, cscGeometry);
	 std::map<std::pair<align::StructureType, align::ID>, Alignable*> alignableStructureMap;
	 recursiveStructureMap(m_alignableMuon->CSCEndcaps(), alignableStructureMap);

	 for (surveyError = cscSurveyErrors->m_surveyErrors.begin();  surveyError != cscSurveyErrors->m_surveyErrors.end();  ++surveyError) {
	    Alignable *alignable = alignableStructureMap[std::pair<align::StructureType, align::ID>(align::StructureType(surveyError->structureType()), surveyError->rawId())];
	    alignable->setSurvey(new SurveyDet(alignable->surface(), surveyError->matrix()));
	 }
      }
      else { // not survey (AlignableRcd)
	 edm::ESHandle<Alignments> cscAlignments;
	 edm::ESHandle<AlignmentErrors> cscAlignmentErrors;
	 iSetup.get<CSCAlignmentRcd>().get(m_CSCAlignmentsLabel, cscAlignments);
	 iSetup.get<CSCAlignmentErrorRcd>().get(m_CSCErrorsLabel, cscAlignmentErrors);
	 aligner.applyAlignments<CSCGeometry>(cscGeometry, &(*cscAlignments), &(*cscAlignmentErrors));

	 m_alignableMuon = new AlignableMuon(dtGeometry, cscGeometry);
	 std::map<std::pair<align::StructureType, align::ID>, Alignable*> alignableStructureMap;
	 recursiveStructureMap(m_alignableMuon->CSCEndcaps(), alignableStructureMap);
	 std::map<align::ID, Alignable*> alignableMap;
	 recursiveMap(m_alignableMuon->CSCEndcaps(), alignableMap, false);

	 for (std::vector<AlignTransformError>::const_iterator alignmentError = cscAlignmentErrors->m_alignError.begin();  alignmentError != cscAlignmentErrors->m_alignError.end();  ++alignmentError) {

	    align::ErrorMatrix matrix6by6 = ROOT::Math::SMatrixIdentity();
	    CLHEP::HepSymMatrix matrix3by3 = alignmentError->matrix();
	    for (int i = 0;  i < 3;  i++) {
	       for (int j = 0;  j < 3;  j++) {
		  matrix6by6(i, j) = matrix3by3(i+1, j+1);
	       }
	    }
	    matrix6by6(3,3) = m_missingErrorAngle;
	    matrix6by6(4,4) = m_missingErrorAngle;
	    matrix6by6(5,5) = m_missingErrorAngle;
	    Alignable *alignable = alignableMap[alignmentError->rawId()];
	    alignable->setSurvey(new SurveyDet(alignable->surface(), matrix6by6));
	 }

	 // get all the ones we missed (non-Det non-DetUnits)
	 for (std::map<std::pair<align::StructureType, align::ID>, Alignable*>::const_iterator iter = alignableStructureMap.begin();  iter != alignableStructureMap.end();  ++iter) {
	    if (iter->second->survey() == NULL) {
	       align::ErrorMatrix matrix6by6 = ROOT::Math::SMatrixIdentity();
	       matrix6by6(0,0) = m_missingErrorTranslation;
	       matrix6by6(1,1) = m_missingErrorTranslation;
	       matrix6by6(2,2) = m_missingErrorTranslation;
	       matrix6by6(3,3) = m_missingErrorAngle;
	       matrix6by6(4,4) = m_missingErrorAngle;
	       matrix6by6(5,5) = m_missingErrorAngle;
	       iter->second->setSurvey(new SurveyDet(iter->second->surface(), matrix6by6));
	    }
	 }

       }
    }
   else if (m_CSCInputMode == std::string("cff")) {
      if (m_CSCInput.getParameter<bool>("survey") != m_CSCInputSurvey)
	 throw cms::Exception("BadConfig") << "CSCInputSurvey = " << (m_CSCInputSurvey ? "true" : "false")
					   << " but CSCInput.survey = " << (m_CSCInput.getParameter<bool>("survey") ? "true" : "false")
					   << " (you probably loaded the wrong .cff)" << std::endl;

      if (!m_CSCInputSurvey) {
	 if (m_CSCInput.getParameter<bool>("eulerAngles") != m_alwaysEulerAngles)
	    throw cms::Exception("BadConfig") << "alwaysEulerAngles = " << (m_alwaysEulerAngles ? "true" : "false")
					      << " but CSCInput.eulerAngles = " << (m_CSCInput.getParameter<bool>("eulerAngles") ? "true" : "false")
					      << " (you probably loaded the wrong .cff)" << std::endl;
      }

      m_alignableMuon = new AlignableMuon(dtGeometry, cscGeometry);      

      recursiveReadIn(m_alignableMuon->CSCEndcaps(), align::PositionType(), align::RotationType(), m_CSCInput, false, m_CSCInputSurvey);
   }
   else assert(false);  // constructor should have caught this




   if (m_CSCOutputMode == std::string("none")) {
      // nothing!
   }
   else if (m_CSCOutputMode == std::string("db")) {
      edm::Service<cond::service::PoolDBOutputService> poolDbService;
      if (!poolDbService.isAvailable()) throw cms::Exception("BadConfig") << "PoolDBOutputService is missing" << std::endl;

      if (m_CSCOutputSurvey) {
	 Alignments* cscAlignments = new Alignments();
	 SurveyErrors* cscSurveyErrors = new SurveyErrors();

	 std::vector<Alignable*> alignableList;
	 recursiveList(m_alignableMuon->CSCEndcaps(), alignableList);
	 for (std::vector<Alignable*>::const_iterator alignable = alignableList.begin();  alignable != alignableList.end();  ++alignable) {
	    if ((*alignable)->alignableObjectId() != align::AlignableDetUnit) {  // don't double-count CSC AlignableDets

	       Alignable *aliid = *alignable;
	       while (aliid->id() == 0) aliid = aliid->components()[0];

	       const align::PositionType& pos = (*alignable)->survey()->position();
	       const align::RotationType& rot = (*alignable)->survey()->rotation();

	       AlignTransform value(CLHEP::Hep3Vector(pos.x(), pos.y(), pos.z()),
				    CLHEP::HepRotation(CLHEP::HepRep3x3(rot.xx(), rot.xy(), rot.xz(),
									rot.yx(), rot.yy(), rot.yz(),
									rot.zx(), rot.zy(), rot.zz())),
				    aliid->id());
	       SurveyError error((*alignable)->alignableObjectId(), aliid->id(), (*alignable)->survey()->errors());

	       cscAlignments->m_align.push_back(value);
	       cscSurveyErrors->m_surveyErrors.push_back(error);
	    }
	 }

	 const std::string cscSurveyRecordName("CSCSurveyRcd");
	 const std::string cscSurveyErrorRecordName("CSCSurveyErrorRcd");
	 if (poolDbService->isNewTagRequest(cscSurveyRecordName)) {
	    poolDbService->createNewIOV<Alignments>(&(*cscAlignments), poolDbService->endOfTime(), cscSurveyRecordName);
	 }
	 else {
	    poolDbService->appendSinceTime<Alignments>(&(*cscAlignments), poolDbService->currentTime(), cscSurveyRecordName);
	 }
	 if (poolDbService->isNewTagRequest(cscSurveyErrorRecordName)) {
	    poolDbService->createNewIOV<SurveyErrors>(&(*cscSurveyErrors), poolDbService->endOfTime(), cscSurveyErrorRecordName);
	 }
	 else {
	    poolDbService->appendSinceTime<SurveyErrors>(&(*cscSurveyErrors), poolDbService->currentTime(), cscSurveyErrorRecordName);
	 }	 
      }
      else {
	 Alignments* cscAlignments = m_alignableMuon->cscAlignments();
	 AlignmentErrors* cscAlignmentErrors = m_alignableMuon->cscAlignmentErrors();

	 const std::string cscAlignRecordName("CSCAlignmentRcd");
	 const std::string cscAlignErrorRecordName("CSCAlignmentErrorRcd");
	 if (poolDbService->isNewTagRequest(cscAlignRecordName)) {
	    poolDbService->createNewIOV<Alignments>(&(*cscAlignments), poolDbService->endOfTime(), cscAlignRecordName);
	 }
	 else {
	    poolDbService->appendSinceTime<Alignments>(&(*cscAlignments), poolDbService->currentTime(), cscAlignRecordName);
	 }
	 if (poolDbService->isNewTagRequest(cscAlignErrorRecordName)) {
	    poolDbService->createNewIOV<AlignmentErrors>(&(*cscAlignmentErrors), poolDbService->endOfTime(), cscAlignErrorRecordName);
	 }
	 else {
	    poolDbService->appendSinceTime<AlignmentErrors>(&(*cscAlignmentErrors), poolDbService->currentTime(), cscAlignErrorRecordName);
	 }
      }
   }
   else if (m_CSCOutputMode == std::string("cff")) {
      // write to ParameterSets
      std::ofstream output(m_CSCOutputFileName.c_str());
      output << "block " << m_CSCOutputBlockName << " = {" << std::endl;
      output << "    PSet CSCInput = {" << std::endl;
      output << "        bool survey = " << (m_CSCOutputSurvey ? "true" : "false") << std::endl;
      if (m_CSCOutputSurvey  ||  m_alwaysEulerAngles) output << "        bool eulerAngles = true" << std::endl;
      else output << "        bool eulerAngles = false" << std::endl;

      recursivePrintOut(m_alignableMuon->CSCEndcaps(), align::PositionType(), align::RotationType(), output, 2, false, m_CSCOutputSurvey);

      output << "    }" << std::endl << "}" << std::endl;
   }
   else assert(false);  // constructor should have caught this
}

void MuonGeometryDBConverter::recursiveReadIn(std::vector<Alignable*> alignables,
					      align::PositionType globalPosition,
					      align::RotationType globalRotation,
					      edm::ParameterSet pset, bool DT, bool survey) {
   static AlignableObjectId converter;

   int i = 0;
   for (std::vector<Alignable*>::const_iterator alignable = alignables.begin();  alignable != alignables.end();  ++alignable) {
      i++;
      std::string name = converter.typeToName((*alignable)->alignableObjectId());
      if (DT) {
	 if ((*alignable)->alignableObjectId() == align::AlignableDet) name = std::string("DTSuperLayer");
	 if ((*alignable)->alignableObjectId() == align::AlignableDetUnit) name = std::string("DTLayer");
      }
      else {
	 if ((*alignable)->alignableObjectId() == align::AlignableDet) name = std::string("CSCLayer");
	 if ((*alignable)->alignableObjectId() == align::AlignableDetUnit) return;
      }

      std::stringstream fullnameSS;
      fullnameSS << name << i;
      std::string fullname;
      fullnameSS >> fullname;
      edm::ParameterSet alipset = pset.getParameter<edm::ParameterSet>(fullname);

      if (survey) {
	 align::PositionType pos(alipset.getParameter<double>("x"), alipset.getParameter<double>("y"), alipset.getParameter<double>("z"));

	 align::EulerAngles eulerAngles(3);
	 eulerAngles(1) = alipset.getParameter<double>("a");
	 eulerAngles(2) = alipset.getParameter<double>("b");
	 eulerAngles(3) = alipset.getParameter<double>("c");
	 align::RotationType rot(align::toMatrix(eulerAngles));

	 std::vector<double> errors1 = alipset.getParameter<std::vector<double> >("errors");
	 ROOT::Math::SVector<double, 21> errors2;
	 for (int j = 0;  j < 21;  j++) errors2(j) = errors1[j];
	 align::ErrorMatrix errors3(errors2);

	 std::cout << "hey " << pos << " " << rot << " " << errors2 << std::endl;

	 align::PositionType position = align::PositionType(globalRotation * pos.basicVector() + globalPosition.basicVector());
	 align::RotationType rotation = globalRotation * rot;

	 // do with them what thou wilt

	 recursiveReadIn((*alignable)->components(), position, rotation, alipset, DT, survey);
      }
      else {
	 if ((*alignable)->alignableObjectId() == align::AlignableDTChamber   ||
	     (*alignable)->alignableObjectId() == align::AlignableCSCChamber  ||
	     (*alignable)->alignableObjectId() == align::AlignableDet         ||
	     (*alignable)->alignableObjectId() == align::AlignableDetUnit) {

	    align::PositionType pos(alipset.getParameter<double>("x"), alipset.getParameter<double>("y"), alipset.getParameter<double>("z")); 
	    align::RotationType rot;
	    if (m_alwaysEulerAngles) {
	       align::EulerAngles eulerAngles(3);
	       eulerAngles(1) = alipset.getParameter<double>("a");
	       eulerAngles(2) = alipset.getParameter<double>("b");
	       eulerAngles(3) = alipset.getParameter<double>("c");
	       align::RotationType rot(align::toMatrix(eulerAngles));
	    }
	    else {
	       double phix = alipset.getParameter<double>("phix");
	       double phiy = alipset.getParameter<double>("phiy");
	       double phiz = alipset.getParameter<double>("phiz");
	    
	       align::RotationType rotX( 1.,         0.,         0.,
					 0.,         cos(phix),  sin(phix),
					 0.,        -sin(phix),  cos(phix));
	       align::RotationType rotY( cos(phiy),  0.,        -sin(phiy), 
					 0.,         1.,         0.,
					 sin(phiy),  0.,         cos(phiy));
	       align::RotationType rotZ( cos(phiz),  sin(phiz),  0.,
					 -sin(phiz),  cos(phiz),  0.,
					 0.,         0.,         1.);
	    
	       rot = rotX * rotY * rotZ;
	    }

	    std::vector<double> errors1 = alipset.getParameter<std::vector<double> >("errors");
	    ROOT::Math::SVector<double, 21> errors2;
	    for (int j = 0;  j < 6;  j++) errors2(j) = errors1[j];
	    errors2(6) = 0.; errors2(7) = 0.; errors2(8) = 0.; errors2(9) = m_missingErrorAngle;
	    errors2(10) = 0.; errors2(11) = 0.; errors2(12) = 0.; errors2(13) = 0.; errors2(14) = m_missingErrorAngle;
	    errors2(15) = 0.; errors2(16) = 0.; errors2(17) = 0.; errors2(18) = 0.; errors2(19) = 0.; errors2(20) = m_missingErrorAngle;
	    align::ErrorMatrix errors3(errors2);

	    std::cout << "you " << pos << " " << rot << " " << errors3 << std::endl;

	    align::PositionType position = align::PositionType(globalRotation * pos.basicVector() + globalPosition.basicVector());
	    align::RotationType rotation = globalRotation * rot;

	    // do with them what thou wilt

	    recursiveReadIn((*alignable)->components(), position, rotation, alipset, DT, survey);
	 }
	 else {
	    recursiveReadIn((*alignable)->components(), globalPosition, globalRotation, alipset, DT, survey);
	 }
      }
   }
}

void MuonGeometryDBConverter::recursivePrintOut(std::vector<Alignable*> alignables,
						align::PositionType globalPosition, align::RotationType globalRotation,
						std::ofstream &output, int depth, bool DT, bool survey) {
   static AlignableObjectId converter;

   int i = 0;
   for (std::vector<Alignable*>::const_iterator alignable = alignables.begin();  alignable != alignables.end();  ++alignable) {
      i++;
      std::string name = converter.typeToName((*alignable)->alignableObjectId());
      if (DT) {
	 if ((*alignable)->alignableObjectId() == align::AlignableDet) name = std::string("DTSuperLayer");
	 if ((*alignable)->alignableObjectId() == align::AlignableDetUnit) name = std::string("DTLayer");
      }
      else {
	 if ((*alignable)->alignableObjectId() == align::AlignableDet) name = std::string("CSCLayer");
	 if ((*alignable)->alignableObjectId() == align::AlignableDetUnit) return;
      }

      for (int d = 0;  d < depth * 4;  d++) output << " ";
      output << "PSet " << name << i << " = {" << std::endl;

      if (survey) {
	 align::PositionType pos = align::PositionType(globalRotation.multiplyInverse((*alignable)->globalPosition().basicVector() - globalPosition.basicVector()));
	 align::RotationType rot = globalRotation.multiplyInverse((*alignable)->globalRotation());

	 for (int d = 0;  d < (depth+1) * 4;  d++) output << " ";
	 output << "double x = " << pos.x() << " double y = " << pos.y() << " double z = " << pos.z() << std::endl;

	 // Standard Euler angles
	 align::EulerAngles eulerAngles = align::toAngles(rot);
	 for (int d = 0;  d < (depth+1) * 4;  d++) output << " ";
	 output << "double a = " << eulerAngles(1) << " double b = " << eulerAngles(2) << " double c = " << eulerAngles(3) << std::endl;

	 align::ErrorMatrix errors = (*alignable)->survey()->errors();
	 for (int d = 0;  d < (depth+1) * 4;  d++) output << " ";
	 output << "vdouble errors = {" << errors(0,0) << ", " << std::endl;
	 for (int d = 0;  d < (depth+1) * 4;  d++) output << " ";
	 output << "                  " << errors(1,0) << ", " << errors(1,1) << ", " << std::endl;
	 for (int d = 0;  d < (depth+1) * 4;  d++) output << " ";
	 output << "                  " << errors(2,0) << ", " << errors(2,1) << ", " << errors(2,2) << ", " << std::endl;
	 for (int d = 0;  d < (depth+1) * 4;  d++) output << " ";
	 output << "                  " << errors(3,0) << ", " << errors(3,1) << ", " << errors(3,2) << ", " << errors(3,3) << ", " << std::endl;
	 for (int d = 0;  d < (depth+1) * 4;  d++) output << " ";
	 output << "                  " << errors(4,0) << ", " << errors(4,1) << ", " << errors(4,2) << ", " << errors(4,3) << ", " << errors(4,4) << ", " << std::endl;
	 for (int d = 0;  d < (depth+1) * 4;  d++) output << " ";
	 output << "                  " << errors(5,0) << ", " << errors(5,1) << ", " << errors(5,2) << ", " << errors(5,3) << ", " << errors(5,4) << ", " << errors(5,5) << "}" << std::endl;

	 recursivePrintOut((*alignable)->components(), (*alignable)->globalPosition(), (*alignable)->globalRotation(), output, depth+1, DT, survey);
      }
      else {
	 if ((*alignable)->alignableObjectId() == align::AlignableDTChamber   ||
	     (*alignable)->alignableObjectId() == align::AlignableCSCChamber  ||
	     (*alignable)->alignableObjectId() == align::AlignableDet         ||
	     (*alignable)->alignableObjectId() == align::AlignableDetUnit) {

	    align::PositionType pos = align::PositionType(globalRotation.multiplyInverse((*alignable)->globalPosition().basicVector() - globalPosition.basicVector()));
	    align::RotationType rot = globalRotation.multiplyInverse((*alignable)->globalRotation());

	    for (int d = 0;  d < (depth+1) * 4;  d++) output << " ";
	    output << "double x = " << pos.x() << " double y = " << pos.y() << " double z = " << pos.z() << std::endl;

	    if (m_alwaysEulerAngles) {
	       align::EulerAngles eulerAngles = align::toAngles(rot);
	       for (int d = 0;  d < (depth+1) * 4;  d++) output << " ";
	       output << "double a = " << eulerAngles(1) << " double b = " << eulerAngles(2) << " double c = " << eulerAngles(3) << std::endl;
	    }
	    else {
	       // alignment angles are non-standard Euler angles (the Z-Y-X convention)
	       double phix = atan2(rot.yz(), rot.zz());
	       double phiy = asin(-rot.xz());
	       double phiz = atan2(rot.xy(), rot.xx());

	       for (int d = 0;  d < (depth+1) * 4;  d++) output << " ";
	       output << "double phix = " << phix << " double phiy = " << phiy << " double phiz = " << phiz << std::endl;
	    }

	    align::ErrorMatrix errors = (*alignable)->survey()->errors();
	    for (int d = 0;  d < (depth+1) * 4;  d++) output << " ";
	    output << "vdouble errors = {" << errors(0,0) << ", " << std::endl;
	    for (int d = 0;  d < (depth+1) * 4;  d++) output << " ";
	    output << "                  " << errors(1,0) << ", " << errors(1,1) << ", " << std::endl;
	    for (int d = 0;  d < (depth+1) * 4;  d++) output << " ";
	    output << "                  " << errors(2,0) << ", " << errors(2,1) << ", " << errors(2,2) << "}" << std::endl;

	    recursivePrintOut((*alignable)->components(), (*alignable)->globalPosition(), (*alignable)->globalRotation(), output, depth+1, DT, survey);
	 }
	 else {
	    recursivePrintOut((*alignable)->components(), globalPosition, globalRotation, output, depth+1, DT, survey);
	 }
      }

      for (int d = 0;  d < depth * 4;  d++) output << " ";
      output << "}" << std::endl;
   }
}


// ------------ method called once each job just after ending the event loop  ------------
void 
MuonGeometryDBConverter::endJob() { }

//define this as a plug-in
DEFINE_FWK_MODULE(MuonGeometryDBConverter);
