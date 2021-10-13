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
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// user include files
#include "Alignment/MuonAlignment/interface/MuonAlignmentInputMethod.h"

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
MuonAlignmentInputMethod::MuonAlignmentInputMethod(edm::ConsumesCollector iC)
    : idealGeometryLabel("idealForInputMethod"),
      dtGeomToken_(iC.esConsumes(edm::ESInputTag("", idealGeometryLabel))),
      cscGeomToken_(iC.esConsumes(edm::ESInputTag("", idealGeometryLabel))),
      gemGeomToken_(iC.esConsumes(edm::ESInputTag("", idealGeometryLabel))) {}
MuonAlignmentInputMethod::MuonAlignmentInputMethod(std::string idealLabel, edm::ConsumesCollector iC)
    : idealGeometryLabel(idealLabel),
      dtGeomToken_(iC.esConsumes(edm::ESInputTag("", idealGeometryLabel))),
      cscGeomToken_(iC.esConsumes(edm::ESInputTag("", idealGeometryLabel))),
      gemGeomToken_(iC.esConsumes(edm::ESInputTag("", idealGeometryLabel))) {}

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
  edm::ESHandle<GEMGeometry> gemGeometry;
  dtGeometry = iSetup.getHandle(dtGeomToken_);
  cscGeometry = iSetup.getHandle(cscGeomToken_);
  gemGeometry = iSetup.getHandle(gemGeomToken_);
  return new AlignableMuon(&(*dtGeometry), &(*cscGeometry), &(*gemGeometry));
}

//
// const member functions
//

//
// static member functions
//
