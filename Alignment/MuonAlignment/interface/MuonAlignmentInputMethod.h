#ifndef Alignment_MuonAlignment_MuonAlignmentInputMethod_h
#define Alignment_MuonAlignment_MuonAlignmentInputMethod_h
// -*- C++ -*-
//
// Package:     MuonAlignment
// Class  :     MuonAlignmentInputMethod
//
/**\class MuonAlignmentInputMethod MuonAlignmentInputMethod.h Alignment/MuonAlignment/interface/MuonAlignmentInputMethod.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Jim Pivarski
//         Created:  Thu Mar  6 14:10:22 CST 2008
// $Id: MuonAlignmentInputMethod.h,v 1.1 2008/03/15 20:26:46 pivarski Exp $
//

// system include files
#include <memory>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
// user include files
#include "Alignment/MuonAlignment/interface/AlignableMuon.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
// forward declarations

class MuonAlignmentInputMethod {
public:
  MuonAlignmentInputMethod();
  MuonAlignmentInputMethod(edm::ConsumesCollector iC);
  MuonAlignmentInputMethod(std::string idealLabel, edm::ConsumesCollector iC);
  virtual ~MuonAlignmentInputMethod();

  // ---------- const member functions ---------------------

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------

  virtual AlignableMuon *newAlignableMuon(const edm::EventSetup &iSetup) const;

  MuonAlignmentInputMethod(const MuonAlignmentInputMethod &) = delete;  // stop default

  const MuonAlignmentInputMethod &operator=(const MuonAlignmentInputMethod &) = delete;  // stop default

private:
  // ---------- member data --------------------------------
  const std::string idealGeometryLabel;

  const edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeomToken_;
  const edm::ESGetToken<CSCGeometry, MuonGeometryRecord> cscGeomToken_;
  const edm::ESGetToken<GEMGeometry, MuonGeometryRecord> gemGeomToken_;
};

#endif
