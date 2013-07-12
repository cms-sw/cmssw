// -*- C++ -*-
//
// Package:    SiStripCommissioningSeedFilter
// Class:      SiStripCommissioningSeedFilter
// 
/**\class SiStripCommissioningSeedFilter SiStripCommissioningSeedFilter.cc myTestArea/SiStripCommissioningSeedFilter/src/SiStripCommissioningSeedFilter.cc

 Description: simply filter acording to the run type

 Implementation:
     Uses information from SiStripEventSummary, so it has to be called after Raw2Digi.
*/
//
// Original Author:  Christophe DELAERE
//         Created:  Fri Jan 18 12:17:46 CET 2008
// $Id: SiStripCommissioningSeedFilter.cc,v 1.2 2008/04/12 20:06:27 delaer Exp $
//
//


// system include files
#include <memory>
#include <algorithm>

// user include files
#include "DQM/SiStripCommissioningSources/interface/SiStripCommissioningSeedFilter.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"


//
// constructors and destructor
//
SiStripCommissioningSeedFilter::SiStripCommissioningSeedFilter(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
   inputModuleLabel_ = iConfig.getParameter<edm::InputTag>( "InputModuleLabel" ) ;
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool
SiStripCommissioningSeedFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   
   edm::Handle<TrajectorySeedCollection> seedcoll;
   iEvent.getByLabel(inputModuleLabel_,seedcoll);
   bool result = (*seedcoll).size()>0;
   return result;
}

