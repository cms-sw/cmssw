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

#include "FWCore/MessageLogger/interface/MessageLogger.h"
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
MuonAlignmentInputMethod::MuonAlignmentInputMethod() : idealGeometryLabel("idealForInputMethod") {}
MuonAlignmentInputMethod::MuonAlignmentInputMethod(std::string idealLabel) : idealGeometryLabel(idealLabel) {}

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

AlignableMuon* MuonAlignmentInputMethod::newAlignableMuon(const edm::EventSetup& iSetup) const {
  edm::ESHandle<DTGeometry> dtGeometry;
  edm::ESHandle<CSCGeometry> cscGeometry;
  iSetup.get<MuonGeometryRecord>().get(idealGeometryLabel, dtGeometry);
  iSetup.get<MuonGeometryRecord>().get(idealGeometryLabel, cscGeometry);
  return new AlignableMuon(&(*dtGeometry), &(*cscGeometry));
}

//
// const member functions
//

//
// static member functions
//
