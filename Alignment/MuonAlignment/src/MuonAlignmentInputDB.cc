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
#include "Geometry/GeometryAligner/interface/GeometryAligner.h"
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
MuonAlignmentInputDB::MuonAlignmentInputDB(const DTGeometry* dtGeometry,
                                           const CSCGeometry* cscGeometry,
                                           const GEMGeometry* gemGeometry,
                                           const Alignments* dtAlignments,
                                           const Alignments* cscAlignments,
                                           const Alignments* gemAlignments,
                                           const Alignments* globalPositionRcd)
    : dtGeometry_(dtGeometry),
      cscGeometry_(cscGeometry),
      gemGeometry_(gemGeometry),
      dtAlignments_(dtAlignments),
      cscAlignments_(cscAlignments),
      gemAlignments_(gemAlignments),
      globalPositionRcd_(globalPositionRcd),
      m_getAPEs(false) {}
MuonAlignmentInputDB::MuonAlignmentInputDB(const DTGeometry* dtGeometry,
                                           const CSCGeometry* cscGeometry,
                                           const GEMGeometry* gemGeometry,
                                           const Alignments* dtAlignments,
                                           const Alignments* cscAlignments,
                                           const Alignments* gemAlignments,
                                           const AlignmentErrorsExtended* dtAlignmentErrorsExtended,
                                           const AlignmentErrorsExtended* cscAlignmentErrorsExtended,
                                           const AlignmentErrorsExtended* gemAlignmentErrorsExtended,
                                           const Alignments* globalPositionRcd)
    : dtGeometry_(dtGeometry),
      cscGeometry_(cscGeometry),
      gemGeometry_(gemGeometry),
      dtAlignments_(dtAlignments),
      cscAlignments_(cscAlignments),
      gemAlignments_(gemAlignments),
      dtAlignmentErrorsExtended_(dtAlignmentErrorsExtended),
      cscAlignmentErrorsExtended_(cscAlignmentErrorsExtended),
      gemAlignmentErrorsExtended_(gemAlignmentErrorsExtended),
      globalPositionRcd_(globalPositionRcd),
      m_getAPEs(true) {}

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

AlignableMuon* MuonAlignmentInputDB::newAlignableMuon() const {
  if (m_getAPEs) {
    GeometryAligner aligner;
    aligner.applyAlignments<DTGeometry>(dtGeometry_,
                                        dtAlignments_,
                                        dtAlignmentErrorsExtended_,
                                        align::DetectorGlobalPosition(*globalPositionRcd_, DetId(DetId::Muon)));
    aligner.applyAlignments<CSCGeometry>(cscGeometry_,
                                         cscAlignments_,
                                         cscAlignmentErrorsExtended_,
                                         align::DetectorGlobalPosition(*globalPositionRcd_, DetId(DetId::Muon)));
    aligner.applyAlignments<GEMGeometry>(gemGeometry_,
                                         gemAlignments_,
                                         gemAlignmentErrorsExtended_,
                                         align::DetectorGlobalPosition(*globalPositionRcd_, DetId(DetId::Muon)));

  } else {
    AlignmentErrorsExtended dtAlignmentErrorsExtended2, cscAlignmentErrorsExtended2, gemAlignmentErrorsExtended2;

    for (std::vector<AlignTransform>::const_iterator i = dtAlignments_->m_align.begin();
         i != dtAlignments_->m_align.end();
         ++i) {
      CLHEP::HepSymMatrix empty_matrix(3, 0);
      AlignTransformErrorExtended empty_error(empty_matrix, i->rawId());
      dtAlignmentErrorsExtended2.m_alignError.push_back(empty_error);
    }
    for (std::vector<AlignTransform>::const_iterator i = cscAlignments_->m_align.begin();
         i != cscAlignments_->m_align.end();
         ++i) {
      CLHEP::HepSymMatrix empty_matrix(3, 0);
      AlignTransformErrorExtended empty_error(empty_matrix, i->rawId());
      cscAlignmentErrorsExtended2.m_alignError.push_back(empty_error);
    }
    for (std::vector<AlignTransform>::const_iterator i = gemAlignments_->m_align.begin();
         i != gemAlignments_->m_align.end();
         ++i) {
      CLHEP::HepSymMatrix empty_matrix(3, 0);
      AlignTransformErrorExtended empty_error(empty_matrix, i->rawId());
      gemAlignmentErrorsExtended2.m_alignError.push_back(empty_error);
    }

    GeometryAligner aligner;
    aligner.applyAlignments<DTGeometry>(dtGeometry_,
                                        dtAlignments_,
                                        &dtAlignmentErrorsExtended2,
                                        align::DetectorGlobalPosition(*globalPositionRcd_, DetId(DetId::Muon)));
    aligner.applyAlignments<CSCGeometry>(cscGeometry_,
                                         cscAlignments_,
                                         &cscAlignmentErrorsExtended2,
                                         align::DetectorGlobalPosition(*globalPositionRcd_, DetId(DetId::Muon)));
    aligner.applyAlignments<GEMGeometry>(gemGeometry_,
                                         gemAlignments_,
                                         &gemAlignmentErrorsExtended2,
                                         align::DetectorGlobalPosition(*globalPositionRcd_, DetId(DetId::Muon)));
  }

  return new AlignableMuon(dtGeometry_, cscGeometry_, gemGeometry_);
}

//
// const member functions
//

//
// static member functions
//
-- dummy change --
