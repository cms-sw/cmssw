// -*- C++ -*-
//
// Package:     MuonAlignment
// Class  :     MuonAlignmentInputSurveyDB
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Jim Pivarski
//         Created:  Thu Mar  6 17:30:46 CST 2008
// $Id: MuonAlignmentInputSurveyDB.cc,v 1.1 2008/03/15 20:26:46 pivarski Exp $
//

// system include files
#include "FWCore/Framework/interface/ESHandle.h"

// user include files
#include "Alignment/MuonAlignment/interface/MuonAlignmentInputSurveyDB.h"
#include "CondFormats/AlignmentRecord/interface/DTSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTSurveyErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCSurveyErrorExtendedRcd.h"
#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MuonAlignmentInputSurveyDB::MuonAlignmentInputSurveyDB()
    : m_dtLabel(""), m_cscLabel(""), idealGeometryLabel("idealForInputSurveyDB") {}

MuonAlignmentInputSurveyDB::MuonAlignmentInputSurveyDB(std::string dtLabel,
                                                       std::string cscLabel,
                                                       std::string idealLabel)
    : m_dtLabel(dtLabel), m_cscLabel(cscLabel), idealGeometryLabel(idealLabel) {}

// MuonAlignmentInputSurveyDB::MuonAlignmentInputSurveyDB(const MuonAlignmentInputSurveyDB& rhs)
// {
//    // do actual copying here;
// }

MuonAlignmentInputSurveyDB::~MuonAlignmentInputSurveyDB() {}

//
// assignment operators
//
// const MuonAlignmentInputSurveyDB& MuonAlignmentInputSurveyDB::operator=(const MuonAlignmentInputSurveyDB& rhs)
// {
//   //An exception safe implementation is
//   MuonAlignmentInputSurveyDB temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

AlignableMuon* MuonAlignmentInputSurveyDB::newAlignableMuon(const edm::EventSetup& iSetup) const {
  edm::ESHandle<DTGeometry> dtGeometry;
  edm::ESHandle<CSCGeometry> cscGeometry;
  edm::ESHandle<GEMGeometry> gemGeometry;
  iSetup.get<MuonGeometryRecord>().get(idealGeometryLabel, dtGeometry);
  iSetup.get<MuonGeometryRecord>().get(idealGeometryLabel, cscGeometry);
  iSetup.get<MuonGeometryRecord>().get(idealGeometryLabel, gemGeometry);

  edm::ESHandle<Alignments> dtSurvey;
  edm::ESHandle<SurveyErrors> dtSurveyError;
  edm::ESHandle<Alignments> cscSurvey;
  edm::ESHandle<SurveyErrors> cscSurveyError;
  iSetup.get<DTSurveyRcd>().get(m_dtLabel, dtSurvey);
  iSetup.get<DTSurveyErrorExtendedRcd>().get(m_dtLabel, dtSurveyError);
  iSetup.get<CSCSurveyRcd>().get(m_cscLabel, cscSurvey);
  iSetup.get<CSCSurveyErrorExtendedRcd>().get(m_cscLabel, cscSurveyError);

  AlignableMuon* output = new AlignableMuon(&(*dtGeometry), &(*cscGeometry), &(*gemGeometry));

  unsigned int theSurveyIndex = 0;
  const Alignments* theSurveyValues = &*dtSurvey;
  const SurveyErrors* theSurveyErrors = &*dtSurveyError;
  const auto& barrels = output->DTBarrel();
  for (const auto& iter : barrels) {
    addSurveyInfo_(iter, &theSurveyIndex, theSurveyValues, theSurveyErrors);
  }

  theSurveyIndex = 0;
  theSurveyValues = &*cscSurvey;
  theSurveyErrors = &*cscSurveyError;
  const auto& endcaps = output->CSCEndcaps();
  for (const auto& iter : endcaps) {
    addSurveyInfo_(iter, &theSurveyIndex, theSurveyValues, theSurveyErrors);
  }

  return output;
}

// This function was copied (with minimal modifications) from
// Alignment/CommonAlignmentProducer/plugins/AlignmentProducer.cc
// (version CMSSW_2_0_0_pre1), guaranteed to work the same way
// unless AlignmentProducer.cc's version changes!
void MuonAlignmentInputSurveyDB::addSurveyInfo_(Alignable* ali,
                                                unsigned int* theSurveyIndex,
                                                const Alignments* theSurveyValues,
                                                const SurveyErrors* theSurveyErrors) const {
  const auto& comp = ali->components();

  unsigned int nComp = comp.size();

  for (unsigned int i = 0; i < nComp; ++i)
    addSurveyInfo_(comp[i], theSurveyIndex, theSurveyValues, theSurveyErrors);

  const SurveyError& error = theSurveyErrors->m_surveyErrors[*theSurveyIndex];

  if (ali->geomDetId().rawId() != error.rawId() || ali->alignableObjectId() != error.structureType()) {
    throw cms::Exception("DatabaseError") << "Error reading survey info from DB. Mismatched id!";
  }

  const CLHEP::Hep3Vector& pos = theSurveyValues->m_align[*theSurveyIndex].translation();
  const CLHEP::HepRotation& rot = theSurveyValues->m_align[*theSurveyIndex].rotation();

  AlignableSurface surf(
      align::PositionType(pos.x(), pos.y(), pos.z()),
      align::RotationType(rot.xx(), rot.xy(), rot.xz(), rot.yx(), rot.yy(), rot.yz(), rot.zx(), rot.zy(), rot.zz()));

  surf.setWidth(ali->surface().width());
  surf.setLength(ali->surface().length());

  ali->setSurvey(new SurveyDet(surf, error.matrix()));

  (*theSurveyIndex)++;
}

//
// const member functions
//

//
// static member functions
//
