// -*- C++ -*-
//
// Package:    SiStripCommissioningBasicPrescaler
// Class:      SiStripCommissioningBasicPrescaler
// 
/**\class SiStripCommissioningBasicPrescaler SiStripCommissioningBasicPrescaler.cc myTestArea/SiStripCommissioningBasicPrescaler/src/SiStripCommissioningBasicPrescaler.cc

 Description: simply filter acording to the run type

 Implementation:
     Uses information from SiStripEventSummary, so it has to be called after Raw2Digi.
*/
//
// Original Author:  Christophe DELAERE
//         Created:  Fri Jan 18 12:17:46 CET 2008
// $Id: SiStripCommissioningBasicPrescaler.cc,v 1.1 2008/10/22 10:44:25 delaer Exp $
//
//


// system include files
#include <memory>

// user include files
#include "DQM/SiStripCommissioningSources/interface/SiStripCommissioningBasicPrescaler.h"

//
// constructors and destructor
//
SiStripCommissioningBasicPrescaler::SiStripCommissioningBasicPrescaler(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
   factor_ = iConfig.getParameter<uint32_t>( "ScaleFactor" ) ;
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool
SiStripCommissioningBasicPrescaler::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   bool result = ((iEvent.id().event()%factor_)==0);
   return result;
}

