// -*- C++ -*-
//
// Package:     MuonAlignment
// Class  :     MuonAlignmentInputDB
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Jim Pivarski
//         Created:  Thu Mar  6 17:30:46 CST 2008
// $Id: MuonAlignmentInputDB.cc,v 1.4 2009/10/07 20:46:39 pivarski Exp $
//

// system include files
#include "FWCore/Framework/interface/ESHandle.h"

// user include files
#include "Alignment/MuonAlignment/interface/MuonAlignmentInputDB.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/GEMAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/GEMAlignmentErrorExtendedRcd.h"
#include "Geometry/CommonTopologies/interface/GeometryAligner.h"
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MuonAlignmentInputDB::MuonAlignmentInputDB()
    : m_dtLabel(""), m_cscLabel(""), m_gemLabel(""), idealGeometryLabel("idealForInputDB"), m_getAPEs(false) {}

MuonAlignmentInputDB::MuonAlignmentInputDB(
    std::string dtLabel, std::string cscLabel, std::string gemLabel, std::string idealLabel, bool getAPEs)
    : m_dtLabel(dtLabel),
      m_cscLabel(cscLabel),
      m_gemLabel(gemLabel),
      idealGeometryLabel(idealLabel),
      m_getAPEs(getAPEs) {}

// MuonAlignmentInputDB::MuonAlignmentInputDB(const MuonAlignmentInputDB& rhs)
// {
//    // do actual copying here;
// }

MuonAlignmentInputDB::~MuonAlignmentInputDB() {}

//
// assignment operators
//
// const MuonAlignmentInputDB& MuonAlignmentInputDB::operator=(const MuonAlignmentInputDB& rhs)
// {
//   //An exception safe implementation is
//   MuonAlignmentInputDB temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

AlignableMuon* MuonAlignmentInputDB::newAlignableMuon(const edm::EventSetup& iSetup) const {
  edm::ESHandle<DTGeometry> dtGeometry;
  edm::ESHandle<CSCGeometry> cscGeometry;
  edm::ESHandle<GEMGeometry> gemGeometry;
  iSetup.get<MuonGeometryRecord>().get(idealGeometryLabel, dtGeometry);
  iSetup.get<MuonGeometryRecord>().get(idealGeometryLabel, cscGeometry);
  iSetup.get<MuonGeometryRecord>().get(idealGeometryLabel, gemGeometry);

  edm::ESHandle<Alignments> dtAlignments;
  edm::ESHandle<AlignmentErrorsExtended> dtAlignmentErrorsExtended;
  edm::ESHandle<Alignments> cscAlignments;
  edm::ESHandle<AlignmentErrorsExtended> cscAlignmentErrorsExtended;
  edm::ESHandle<Alignments> gemAlignments;
  edm::ESHandle<AlignmentErrorsExtended> gemAlignmentErrorsExtended;
  edm::ESHandle<Alignments> globalPositionRcd;

  iSetup.get<DTAlignmentRcd>().get(m_dtLabel, dtAlignments);
  iSetup.get<CSCAlignmentRcd>().get(m_cscLabel, cscAlignments);
  iSetup.get<GEMAlignmentRcd>().get(m_gemLabel, gemAlignments);
  iSetup.get<GlobalPositionRcd>().get(globalPositionRcd);

  if (m_getAPEs) {
    iSetup.get<DTAlignmentErrorExtendedRcd>().get(m_dtLabel, dtAlignmentErrorsExtended);
    iSetup.get<CSCAlignmentErrorExtendedRcd>().get(m_cscLabel, cscAlignmentErrorsExtended);
    iSetup.get<GEMAlignmentErrorExtendedRcd>().get(m_gemLabel, gemAlignmentErrorsExtended);

    GeometryAligner aligner;
    aligner.applyAlignments<DTGeometry>(&(*dtGeometry),
                                        &(*dtAlignments),
                                        &(*dtAlignmentErrorsExtended),
                                        align::DetectorGlobalPosition(*globalPositionRcd, DetId(DetId::Muon)));
    aligner.applyAlignments<CSCGeometry>(&(*cscGeometry),
                                         &(*cscAlignments),
                                         &(*cscAlignmentErrorsExtended),
                                         align::DetectorGlobalPosition(*globalPositionRcd, DetId(DetId::Muon)));
    aligner.applyAlignments<GEMGeometry>(&(*gemGeometry),
                                         &(*gemAlignments),
                                         &(*gemAlignmentErrorsExtended),
                                         align::DetectorGlobalPosition(*globalPositionRcd, DetId(DetId::Muon)));

  } else {
    AlignmentErrorsExtended dtAlignmentErrorsExtended2, cscAlignmentErrorsExtended2, gemAlignmentErrorsExtended2;

    for (std::vector<AlignTransform>::const_iterator i = dtAlignments->m_align.begin();
         i != dtAlignments->m_align.end();
         ++i) {
      CLHEP::HepSymMatrix empty_matrix(3, 0);
      AlignTransformErrorExtended empty_error(empty_matrix, i->rawId());
      dtAlignmentErrorsExtended2.m_alignError.push_back(empty_error);
    }
    for (std::vector<AlignTransform>::const_iterator i = cscAlignments->m_align.begin();
         i != cscAlignments->m_align.end();
         ++i) {
      CLHEP::HepSymMatrix empty_matrix(3, 0);
      AlignTransformErrorExtended empty_error(empty_matrix, i->rawId());
      cscAlignmentErrorsExtended2.m_alignError.push_back(empty_error);
    }
    for (std::vector<AlignTransform>::const_iterator i = gemAlignments->m_align.begin();
         i != gemAlignments->m_align.end();
         ++i) {
      CLHEP::HepSymMatrix empty_matrix(3, 0);
      AlignTransformErrorExtended empty_error(empty_matrix, i->rawId());
      gemAlignmentErrorsExtended2.m_alignError.push_back(empty_error);
    }

    GeometryAligner aligner;
    aligner.applyAlignments<DTGeometry>(&(*dtGeometry),
                                        &(*dtAlignments),
                                        &(dtAlignmentErrorsExtended2),
                                        align::DetectorGlobalPosition(*globalPositionRcd, DetId(DetId::Muon)));
    aligner.applyAlignments<CSCGeometry>(&(*cscGeometry),
                                         &(*cscAlignments),
                                         &(cscAlignmentErrorsExtended2),
                                         align::DetectorGlobalPosition(*globalPositionRcd, DetId(DetId::Muon)));
    aligner.applyAlignments<GEMGeometry>(&(*gemGeometry),
                                         &(*gemAlignments),
                                         &(gemAlignmentErrorsExtended2),
                                         align::DetectorGlobalPosition(*globalPositionRcd, DetId(DetId::Muon)));
  }

  return new AlignableMuon(&(*dtGeometry), &(*cscGeometry), &(*gemGeometry));
}

//
// const member functions
//

//
// static member functions
//
