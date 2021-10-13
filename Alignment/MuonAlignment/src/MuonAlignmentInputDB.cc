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
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
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
MuonAlignmentInputDB::MuonAlignmentInputDB(edm::ConsumesCollector iC)
    : m_dtLabel(""),
      m_cscLabel(""),
      m_gemLabel(""),
      idealGeometryLabel("idealForInputDB"),
      m_getAPEs(false),
      dtGeomToken_(iC.esConsumes(edm::ESInputTag("", idealGeometryLabel))),
      cscGeomToken_(iC.esConsumes(edm::ESInputTag("", idealGeometryLabel))),
      gemGeomToken_(iC.esConsumes(edm::ESInputTag("", idealGeometryLabel))),
      dtAliToken_(iC.esConsumes()),
      cscAliToken_(iC.esConsumes()),
      gemAliToken_(iC.esConsumes()),
      dtAliErrToken_(iC.esConsumes()),
      cscAliErrToken_(iC.esConsumes()),
      gemAliErrToken_(iC.esConsumes()),
      gprToken_(iC.esConsumes()) {}

MuonAlignmentInputDB::MuonAlignmentInputDB(std::string dtLabel,
                                           std::string cscLabel,
                                           std::string gemLabel,
                                           std::string idealLabel,
                                           bool getAPEs,
                                           edm::ConsumesCollector iC)
    : m_dtLabel(dtLabel),
      m_cscLabel(cscLabel),
      m_gemLabel(gemLabel),
      idealGeometryLabel(idealLabel),
      m_getAPEs(getAPEs),
      dtGeomToken_(iC.esConsumes(edm::ESInputTag("", idealGeometryLabel))),
      cscGeomToken_(iC.esConsumes(edm::ESInputTag("", idealGeometryLabel))),
      gemGeomToken_(iC.esConsumes(edm::ESInputTag("", idealGeometryLabel))),
      dtAliToken_(iC.esConsumes()),
      cscAliToken_(iC.esConsumes()),
      gemAliToken_(iC.esConsumes()),
      dtAliErrToken_(iC.esConsumes()),
      cscAliErrToken_(iC.esConsumes()),
      gemAliErrToken_(iC.esConsumes()),
      gprToken_(iC.esConsumes()) {}
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
  edm::ESHandle<Alignments> dtAlignments;
  edm::ESHandle<AlignmentErrorsExtended> dtAlignmentErrorsExtended;
  edm::ESHandle<Alignments> cscAlignments;
  edm::ESHandle<AlignmentErrorsExtended> cscAlignmentErrorsExtended;
  edm::ESHandle<Alignments> gemAlignments;
  edm::ESHandle<AlignmentErrorsExtended> gemAlignmentErrorsExtended;
  edm::ESHandle<Alignments> globalPositionRcd;
  dtGeometry = iSetup.getHandle(dtGeomToken_);
  cscGeometry = iSetup.getHandle(cscGeomToken_);
  gemGeometry = iSetup.getHandle(gemGeomToken_);
  dtAlignments = iSetup.getHandle(dtAliToken_);
  cscAlignments = iSetup.getHandle(cscAliToken_);
  gemAlignments = iSetup.getHandle(gemAliToken_);
  globalPositionRcd = iSetup.getHandle(gprToken_);

  if (m_getAPEs) {
    dtAlignmentErrorsExtended = iSetup.getHandle(dtAliErrToken_);
    cscAlignmentErrorsExtended = iSetup.getHandle(cscAliErrToken_);
    gemAlignmentErrorsExtended = iSetup.getHandle(gemAliErrToken_);

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
