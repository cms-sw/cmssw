// -*- C++ -*-
//
// Package:     MuonAlignment
// Class  :     MuonAlignmentInputMethod
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Jim Pivarski
//         Created:  Thu Mar  6 14:25:07 CST 2008
// $Id: MuonAlignmentInputMethod.cc,v 1.4 2010/05/14 21:07:59 pivarski Exp $
//

// system include files
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"

// user include files
#include "Alignment/MuonAlignment/interface/MuonAlignmentInputMethod.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h"
#include "Geometry/CSCGeometryBuilder/src/CSCGeometryBuilderFromDDD.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MuonAlignmentInputMethod::MuonAlignmentInputMethod() {}

// MuonAlignmentInputMethod::MuonAlignmentInputMethod(const MuonAlignmentInputMethod& rhs)
// {
//    // do actual copying here;
// }

MuonAlignmentInputMethod::~MuonAlignmentInputMethod() {}

//
// assignment operators
//
// const MuonAlignmentInputMethod& MuonAlignmentInputMethod::operator=(const MuonAlignmentInputMethod& rhs)
// {
//   //An exception safe implementation is
//   MuonAlignmentInputMethod temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

AlignableMuon *MuonAlignmentInputMethod::newAlignableMuon(const edm::EventSetup& iSetup) const {
   boost::shared_ptr<DTGeometry> dtGeometry = idealDTGeometry(iSetup);
   boost::shared_ptr<CSCGeometry> cscGeometry = idealCSCGeometry(iSetup);

   return new AlignableMuon(&(*dtGeometry), &(*cscGeometry));
}

boost::shared_ptr<DTGeometry> MuonAlignmentInputMethod::idealDTGeometry(const edm::EventSetup& iSetup) const {
   edm::ESTransientHandle<DDCompactView> cpv;
   iSetup.get<IdealGeometryRecord>().get(cpv);

   edm::ESHandle<MuonDDDConstants> mdc;
   iSetup.get<MuonNumberingRecord>().get(mdc);
   DTGeometryBuilderFromDDD DTGeometryBuilder;

   boost::shared_ptr<DTGeometry> boost_dtGeometry(new DTGeometry );
   DTGeometryBuilder.build(boost_dtGeometry, &(*cpv), *mdc);

   return boost_dtGeometry;
}

boost::shared_ptr<CSCGeometry> MuonAlignmentInputMethod::idealCSCGeometry(const edm::EventSetup& iSetup) const {
   edm::ESTransientHandle<DDCompactView> cpv;
   iSetup.get<IdealGeometryRecord>().get(cpv);

   edm::ESHandle<MuonDDDConstants> mdc;
   iSetup.get<MuonNumberingRecord>().get(mdc);
   CSCGeometryBuilderFromDDD CSCGeometryBuilder;

   boost::shared_ptr<CSCGeometry> boost_cscGeometry(new CSCGeometry);
   CSCGeometryBuilder.build(boost_cscGeometry, &(*cpv), *mdc);

   return boost_cscGeometry;
}

//
// const member functions
//

//
// static member functions
//
