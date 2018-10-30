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
// $Id: MuonAlignmentInputMethod.cc,v 1.3 2009/01/19 11:07:37 flucke Exp $
//

// system include files
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"

// user include files
#include "Alignment/MuonAlignment/interface/MuonAlignmentInputMethod.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h"
#include "Geometry/CSCGeometryBuilder/src/CSCGeometryBuilderFromDDD.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
   std::shared_ptr<DTGeometry> dtGeometry = idealDTGeometry(iSetup);
   std::shared_ptr<CSCGeometry> cscGeometry = idealCSCGeometry(iSetup);

   return new AlignableMuon(&(*dtGeometry), &(*cscGeometry));
}

std::shared_ptr<DTGeometry> MuonAlignmentInputMethod::idealDTGeometry(const edm::EventSetup& iSetup) const {
   edm::ESTransientHandle<DDCompactView> cpv;
   iSetup.get<IdealGeometryRecord>().get(cpv);

   edm::ESHandle<MuonDDDConstants> mdc;
   iSetup.get<MuonNumberingRecord>().get(mdc);
   DTGeometryBuilderFromDDD DTGeometryBuilder;

   auto boost_dtGeometry = std::make_shared<DTGeometry>();
   DTGeometryBuilder.build(*boost_dtGeometry, &(*cpv), *mdc);

   return boost_dtGeometry;
}

std::shared_ptr<CSCGeometry> MuonAlignmentInputMethod::idealCSCGeometry(const edm::EventSetup& iSetup) const {
   edm::ESTransientHandle<DDCompactView> cpv;
   iSetup.get<IdealGeometryRecord>().get(cpv);

   edm::ESHandle<MuonDDDConstants> mdc;
   iSetup.get<MuonNumberingRecord>().get(mdc);
   CSCGeometryBuilderFromDDD CSCGeometryBuilder;

   auto boost_cscGeometry = std::make_shared<CSCGeometry>();
   CSCGeometryBuilder.build(*boost_cscGeometry, &(*cpv), *mdc);

   return boost_cscGeometry;
}

//
// const member functions
//

//
// static member functions
//
