// -*- C++ -*-
//
// Package:    HLTL1NumberFilter
// Class:      HLTL1NumberFilter
// 
/**\class HLTL1NumberFilter HLTL1NumberFilter.cc filter/HLTL1NumberFilter/src/HLTL1NumberFilter.cc

Description: 

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Martin Grunewald
//         Created:  Tue Jan 22 13:55:00 CET 2008
// $Id: HLTL1NumberFilter.cc,v 1.1 2009/08/21 08:11:23 bdahmes Exp $
//
//


// system include files
#include <string>
#include <iostream>
#include <memory>

// user include files
#include "HLTrigger/special/interface/HLTL1NumberFilter.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

//
// constructors and destructor
//
HLTL1NumberFilter::HLTL1NumberFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) 
{
  //now do what ever initialization is needed
  input_  = iConfig.getParameter<edm::InputTag>("rawInput") ;   
  period_ = iConfig.getParameter<unsigned int>("period") ;
  invert_ = iConfig.getParameter<bool>("invert") ;
}


HLTL1NumberFilter::~HLTL1NumberFilter()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
HLTL1NumberFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{
  using namespace edm;

  if (iEvent.isRealData()) {
    bool accept(false);
    edm::Handle<FEDRawDataCollection> theRaw ;
    iEvent.getByLabel(input_,theRaw) ;
    const FEDRawData& data = theRaw->FEDData(FEDNumbering::MINTriggerGTPFEDID) ;
    FEDHeader header(data.data()) ;
    if (period_!=0) accept = ( ( (header.lvl1ID())%period_ ) == 0 );
    if (invert_) accept = !accept;
    return accept;
  } else {
    return true;
  }

}
