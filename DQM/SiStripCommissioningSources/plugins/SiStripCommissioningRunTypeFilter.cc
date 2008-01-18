// -*- C++ -*-
//
// Package:    SiStripCommissioningRunTypeFilter
// Class:      SiStripCommissioningRunTypeFilter
// 
/**\class SiStripCommissioningRunTypeFilter SiStripCommissioningRunTypeFilter.cc myTestArea/SiStripCommissioningRunTypeFilter/src/SiStripCommissioningRunTypeFilter.cc

 Description: simply filter acording to the run type

 Implementation:
     Uses information from SiStripEventSummary, so it has to be called after Raw2Digi.
*/
//
// Original Author:  Christophe DELAERE
//         Created:  Fri Jan 18 12:17:46 CET 2008
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "DQM/SiStripCommissioningSources/interface/SiStripCommissioningRunTypeFilter.h"

//
// constructors and destructor
//
SiStripCommissioningRunTypeFilter::SiStripCommissioningRunTypeFilter(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
   inputModuleLabel_ = iConfig.getParameter<edm::InputTag>( "InputModuleLabel" ) ;
   runType_ = SiStripEnumsAndStrings::runType(iConfig.getParameter<std::string>("runType"));
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool
SiStripCommissioningRunTypeFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   // Retrieve commissioning information from "event summary"
   edm::Handle<SiStripEventSummary> summary;
   iEvent.getByLabel( inputModuleLabel_, summary );
   return (summary->runType() == runType_ );
}

