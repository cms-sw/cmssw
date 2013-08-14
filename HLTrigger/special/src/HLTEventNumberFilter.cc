// -*- C++ -*-
//
// Package:    HLTEventNumberFilter
// Class:      HLTEventNumberFilter
// 
/**\class HLTEventNumberFilter HLTEventNumberFilter.cc filter/HLTEventNumberFilter/src/HLTEventNumberFilter.cc

Description: 

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Martin Grunewald
//         Created:  Tue Jan 22 13:55:00 CET 2008
// $Id: HLTEventNumberFilter.cc,v 1.3 2012/01/21 15:00:16 fwyzard Exp $
//
//


// system include files
#include <string>
#include <iostream>
#include <memory>

// user include files
#include "HLTrigger/special/interface/HLTEventNumberFilter.h"

//
// constructors and destructor
//
HLTEventNumberFilter::HLTEventNumberFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) 
{
  //now do what ever initialization is needed

  period_ = iConfig.getParameter<unsigned int>("period") ;
  invert_ = iConfig.getParameter<bool>("invert") ;
}


HLTEventNumberFilter::~HLTEventNumberFilter()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
HLTEventNumberFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{
  using namespace edm;

  if (iEvent.isRealData()) {
    bool accept(false);
    if (period_!=0) accept = ( ( (iEvent.id().event())%period_ ) == 0 );
    if (invert_) accept = !accept;
    return accept;
  } else {
    return true;
  }

}
