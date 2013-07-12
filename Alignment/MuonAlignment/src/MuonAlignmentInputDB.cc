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
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentErrorRcd.h"
#include "Geometry/TrackingGeometryAligner/interface/GeometryAligner.h"
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
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
MuonAlignmentInputDB::MuonAlignmentInputDB()
  : m_dtLabel(""), m_cscLabel(""), m_getAPEs(false) {}

MuonAlignmentInputDB::MuonAlignmentInputDB(std::string dtLabel, std::string cscLabel, bool getAPEs)
   : m_dtLabel(dtLabel), m_cscLabel(cscLabel), m_getAPEs(getAPEs) {}

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

AlignableMuon *MuonAlignmentInputDB::newAlignableMuon(const edm::EventSetup& iSetup) const {
   boost::shared_ptr<DTGeometry> dtGeometry = idealDTGeometry(iSetup);
   boost::shared_ptr<CSCGeometry> cscGeometry = idealCSCGeometry(iSetup);

   edm::ESHandle<Alignments> dtAlignments;
   edm::ESHandle<AlignmentErrors> dtAlignmentErrors;
   edm::ESHandle<Alignments> cscAlignments;
   edm::ESHandle<AlignmentErrors> cscAlignmentErrors;
   edm::ESHandle<Alignments> globalPositionRcd;

   iSetup.get<DTAlignmentRcd>().get(m_dtLabel, dtAlignments);
   iSetup.get<CSCAlignmentRcd>().get(m_cscLabel, cscAlignments);
   iSetup.get<GlobalPositionRcd>().get(globalPositionRcd);

   if (m_getAPEs) {
      iSetup.get<DTAlignmentErrorRcd>().get(m_dtLabel, dtAlignmentErrors);
      iSetup.get<CSCAlignmentErrorRcd>().get(m_cscLabel, cscAlignmentErrors);

      GeometryAligner aligner;
      aligner.applyAlignments<DTGeometry>(&(*dtGeometry), &(*dtAlignments), &(*dtAlignmentErrors),
					  align::DetectorGlobalPosition(*globalPositionRcd, DetId(DetId::Muon)));
      aligner.applyAlignments<CSCGeometry>(&(*cscGeometry), &(*cscAlignments), &(*cscAlignmentErrors),
					   align::DetectorGlobalPosition(*globalPositionRcd, DetId(DetId::Muon)));
   }
   else {
      AlignmentErrors dtAlignmentErrors2, cscAlignmentErrors2;

      for (std::vector<AlignTransform>::const_iterator i = dtAlignments->m_align.begin();  i != dtAlignments->m_align.end();  ++i) {
	 CLHEP::HepSymMatrix empty_matrix(3, 0);
	 AlignTransformError empty_error(empty_matrix, i->rawId());
	 dtAlignmentErrors2.m_alignError.push_back(empty_error);
      }
      for (std::vector<AlignTransform>::const_iterator i = cscAlignments->m_align.begin();  i != cscAlignments->m_align.end();  ++i) {
	 CLHEP::HepSymMatrix empty_matrix(3, 0);
	 AlignTransformError empty_error(empty_matrix, i->rawId());
	 cscAlignmentErrors2.m_alignError.push_back(empty_error);
      }

      GeometryAligner aligner;
      aligner.applyAlignments<DTGeometry>(&(*dtGeometry), &(*dtAlignments), &(dtAlignmentErrors2),
					  align::DetectorGlobalPosition(*globalPositionRcd, DetId(DetId::Muon)));
      aligner.applyAlignments<CSCGeometry>(&(*cscGeometry), &(*cscAlignments), &(cscAlignmentErrors2),
					   align::DetectorGlobalPosition(*globalPositionRcd, DetId(DetId::Muon)));
   }

   return new AlignableMuon(&(*dtGeometry), &(*cscGeometry));
}

//
// const member functions
//

//
// static member functions
//
