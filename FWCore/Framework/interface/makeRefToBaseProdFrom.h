#ifndef FWCore_Framework_makeRefToBaseProdFrom_h
#define FWCore_Framework_makeRefToBaseProdFrom_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     makeRefToBaseProdFrom
// 
/**\class makeRefToBaseProdFrom makeRefToBaseProdFrom.h "FWCore/Framework/interface/makeRefToBaseProdFrom.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Tue, 02 Dec 2014 16:00:08 GMT
//

// system include files

// user include files
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/RefToBaseProd.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"

// forward declarations

namespace edm {
  template<typename T>
    RefToBaseProd<T> makeRefToBaseProdFrom(RefToBase<T> const& iRef, Event const& iEvent) {
    Handle<View<T>> view;
    iEvent.get(iRef.id(),view);

    return RefToBaseProd<T>(view);
  }
}
#endif
