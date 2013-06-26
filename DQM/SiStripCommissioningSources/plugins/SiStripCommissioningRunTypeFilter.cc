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
// $Id: SiStripCommissioningRunTypeFilter.cc,v 1.2 2008/04/12 20:06:27 delaer Exp $
//
//


// system include files
#include <memory>
#include <algorithm>

// user include files
#include "DQM/SiStripCommissioningSources/interface/SiStripCommissioningRunTypeFilter.h"

//
// constructors and destructor
//
SiStripCommissioningRunTypeFilter::SiStripCommissioningRunTypeFilter(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
   inputModuleLabel_ = iConfig.getParameter<edm::InputTag>( "InputModuleLabel" ) ;
   std::vector<std::string> runTypes = iConfig.getParameter<std::vector<std::string> >("runTypes");
   for(std::vector<std::string>::const_iterator run = runTypes.begin(); run != runTypes.end(); ++run) {
     runTypes_.push_back(SiStripEnumsAndStrings::runType(*run));
   }
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
   return (std::find(runTypes_.begin(),runTypes_.end(),summary->runType())!=runTypes_.end());
}

