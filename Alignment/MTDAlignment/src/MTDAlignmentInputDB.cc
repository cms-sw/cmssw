// -*- C++ -*-
//
// Package:     MTDAlignment
// Class  :     MTDAlignmentInputDB
//

// system include files
#include "FWCore/Framework/interface/ESHandle.h"

// user include files
#include "Alignment/MTDAlignment/interface/MTDAlignmentInputDB.h"
#include "CondFormats/AlignmentRecord/interface/MTDAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/MTDAlignmentErrorExtendedRcd.h"
#include "Geometry/GeometryAligner/interface/GeometryAligner.h"
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "Geometry/Records/interface/MTDGeometryRecord.h"
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
MTDAlignmentInputDB::MTDAlignmentInputDB(const MTDGeometry* mtdGeometry,
                                         const Alignments* mtdAlignments,
                                         const Alignments* globalPositionRcd)
    : mtdGeometry_(mtdGeometry),
      mtdAlignments_(mtdAlignments),
      globalPositionRcd_(globalPositionRcd),
      m_getAPEs(false) {}

MTDAlignmentInputDB::MTDAlignmentInputDB(const MTDGeometry* mtdGeometry,
                                         const Alignments* mtdAlignments,
                                         const AlignmentErrorsExtended* mtdAlignmentErrorsExtended,
                                         const Alignments* globalPositionRcd)
    : mtdGeometry_(mtdGeometry),
      mtdAlignments_(mtdAlignments),
      mtdAlignmentErrorsExtended_(mtdAlignmentErrorsExtended),
      globalPositionRcd_(globalPositionRcd),
      m_getAPEs(true) {}

MTDAlignmentInputDB::~MTDAlignmentInputDB() {}

AlignableMTD* MTDAlignmentInputDB::newAlignableMTD() const {
  if (m_getAPEs) {
    GeometryAligner aligner;
    aligner.applyAlignments<MTDGeometry>(mtdGeometry_,
                                         mtdAlignments_,
                                         mtdAlignmentErrorsExtended_,
                                         align::DetectorGlobalPosition(*globalPositionRcd_, DetId(DetId::Forward)));
  } else {
    AlignmentErrorsExtended mtdAlignmentErrorsExtended2;

    for (std::vector<AlignTransform>::const_iterator i = mtdAlignments_->m_align.begin();
         i != mtdAlignments_->m_align.end();
         ++i) {
      CLHEP::HepSymMatrix empty_matrix(3, 0);
      AlignTransformErrorExtended empty_error(empty_matrix, i->rawId());
      mtdAlignmentErrorsExtended2.m_alignError.push_back(empty_error);
    }

    GeometryAligner aligner;
    aligner.applyAlignments<MTDGeometry>(mtdGeometry_,
                                         mtdAlignments_,
                                         &mtdAlignmentErrorsExtended2,
                                         align::DetectorGlobalPosition(*globalPositionRcd_, DetId(DetId::Forward)));
  }

  return new AlignableMTD(mtdGeometry_);
}

//
// const member functions
//

//
// static member functions
//
