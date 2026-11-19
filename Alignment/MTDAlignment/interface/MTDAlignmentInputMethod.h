#ifndef Alignment_MTDAlignment_MTDAlignmentInputMethod_h
#define Alignment_MTDAlignment_MTDAlignmentInputMethod_h
// -*- C++ -*-
//
// Package:     MTDAlignment
// Class  :     MTDAlignmentInputMethod
//
/**\class MTDAlignmentInputMethod MTDAlignmentInputMethod.h Alignment/MTDAlignment/interface/MTDAlignmentInputMethod.h

 Description: <one line class summary>

 Usage:
    <usage>

*/

// system include files
#include <memory>

#include "FWCore/Framework/interface/EventSetup.h"

// user include files
#include "Alignment/MTDAlignment/interface/AlignableMTD.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"

// forward declarations

class MTDAlignmentInputMethod {
public:
  MTDAlignmentInputMethod();
  MTDAlignmentInputMethod(const MTDGeometry* mtdGeometry);
  virtual ~MTDAlignmentInputMethod();

  // ---------- const member functions ---------------------

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------

  virtual AlignableMTD* newAlignableMTD() const;

  MTDAlignmentInputMethod(const MTDAlignmentInputMethod&) = delete;  // stop default

  const MTDAlignmentInputMethod& operator=(const MTDAlignmentInputMethod&) = delete;  // stop default

private:
  // ---------- member data --------------------------------
  const MTDGeometry* mtdGeometry_;
};

#endif
