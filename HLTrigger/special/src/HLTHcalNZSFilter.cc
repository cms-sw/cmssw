// -*- C++ -*-
//
// Package:    HLTHcalNZSFilter
// Class:      HLTHcalNZSFilter
// 
/**\class HLTHcalNZSFilter HLTHcalNZSFilter.cc filter/HLTHcalNZSFilter/src/HLTHcalNZSFilter.cc

Description: Filter to select HCAL abort gap events

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Bryan DAHMES
//         Created:  Tue Jan 22 13:55:00 CET 2008
// $Id: HLTHcalNZSFilter.cc,v 1.5 2009/07/08 13:26:31 gruen Exp $
//
//


// system include files
#include <string>
#include <iostream>
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/HcalDigi/interface/HcalCalibrationEventTypes.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"

#include "HLTrigger/special/interface/HLTHcalNZSFilter.h"

//
// constructors and destructor
//
HLTHcalNZSFilter::HLTHcalNZSFilter(const edm::ParameterSet& iConfig)
{
  //now do what ever initialization is needed

  DataLabel_  = iConfig.getParameter<std::string>("InputLabel") ;
  Summary_    = iConfig.getUntrackedParameter<bool>("FilterSummary",false) ;
}


HLTHcalNZSFilter::~HLTHcalNZSFilter()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
HLTHcalNZSFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  edm::Handle<FEDRawDataCollection> rawdata;  
  iEvent.getByLabel(DataLabel_,rawdata);

  bool hcalIsZS = false ; 
  for (int i=FEDNumbering::MINHCALFEDID; i<=FEDNumbering::MAXHCALFEDID; i++) {
      const FEDRawData& fedData = rawdata->FEDData(i) ; 
      if ( fedData.size() < 24 ) continue ; 

      // Check for Zero-suppression
      HcalHTRData htr;
      const HcalDCCHeader* dccHeader = (const HcalDCCHeader*)(fedData.data()) ; 
      int nZS = 0 ; bool isZS = false ; 
      for (int spigot=0; spigot<HcalDCCHeader::SPIGOT_COUNT; spigot++) {    
          if (!dccHeader->getSpigotPresent(spigot)) continue;
          
          // Load the given decoder with the pointer and length from this spigot.
          dccHeader->getSpigotData(spigot,htr, fedData.size()); 
          
          // check min length, correct wordcount, empty event, or total length if histo event.
          if ( !htr.isUnsuppressed() ) { isZS = true ; hcalIsZS = true ; nZS++ ; }
      }
      if ( hcalIsZS && !isZS )
          LogWarning("HLTHcalNZSFilter") << "HCAL is ZS, but HCAL FED " << i << " is not" ; 
      if ( isZS && nZS != HcalDCCHeader::SPIGOT_COUNT )
          LogDebug("HLTHcalNZSFilter") << nZS << " out of "
                                       << HcalDCCHeader::SPIGOT_COUNT << " HCAL HTRs for FED "
                                       << i << " are zero-suppressed for this event" ;
  }

  if ( !hcalIsZS ) eventsNZS_++ ; 
  return ( !hcalIsZS ) ; 
}

// ------------ method called once each job just before starting event loop  ------------
void 
HLTHcalNZSFilter::beginJob(const edm::EventSetup&)
{
  eventsNZS_ = 0 ; 
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HLTHcalNZSFilter::endJob() {
  if ( Summary_ ) edm::LogWarning("HLTHcalNZSFilter") << "Kept " << eventsNZS_ << " non-ZS events" ;  
}
