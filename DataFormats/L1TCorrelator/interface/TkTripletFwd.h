#ifndef DataFormatsL1TCorrelator_TkTripletFwd_h
#define DataFormatsL1TCorrelator_TkTripletFwd_h

// Original author: G Karathanasis,
//                    georgios.karathanasis@cern.ch, CU Boulder
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     TkTripletFwd
//

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace l1t {

  namespace io_v1 {
    class TkTriplet;
  }
  using TkTriplet = io_v1::TkTriplet;

  typedef edm::RefProd<TkTriplet> TkTripletRefProd;

  typedef std::vector<TkTriplet> TkTripletCollection;

  typedef edm::Ref<TkTripletCollection> TkTripletRef;
  typedef edm::RefVector<TkTripletCollection> TkTripletRefVector;
  typedef std::vector<TkTripletRef> TkTripletVectorRef;
}  // namespace l1t

#endif
