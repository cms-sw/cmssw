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

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/HcalDigi/interface/HcalCalibrationEventTypes.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"

#include "HLTrigger/special/interface/HLTHcalNZSFilter.h"

//
// constructors and destructor
//
HLTHcalNZSFilter::HLTHcalNZSFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig)
{
  //now do what ever initialization is needed

  dataInputTag_ = iConfig.getParameter<edm::InputTag>("InputTag") ;
  dataInputToken_ = consumes<FEDRawDataCollection>(dataInputTag_);
}


HLTHcalNZSFilter::~HLTHcalNZSFilter()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}

void
HLTHcalNZSFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("InputTag",edm::InputTag("source"));
  descriptions.add("hltHcalNZSFilter",desc);
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool
HLTHcalNZSFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{
  using namespace edm;

  // MC treatment for this filter(NZS not fully emulated in HTR for MC)
  if (!iEvent.isRealData()) return false;

  edm::Handle<FEDRawDataCollection> rawdata;
  iEvent.getByToken(dataInputToken_,rawdata);

  int nFEDs = 0 ; int nNZSfed = 0 ; int nZSfed = 0 ;
  for (int i=FEDNumbering::MINHCALFEDID; i<=FEDNumbering::MAXHCALFEDID; i++) {
      const FEDRawData& fedData = rawdata->FEDData(i) ;
      if ( fedData.size() < 24 ) continue ;
      nFEDs++ ;

      // Check for Zero-suppression
      HcalHTRData htr;
      const HcalDCCHeader* dccHeader = (const HcalDCCHeader*)(fedData.data()) ;
      int nZS = 0 ; int nUS = 0 ; int nSpigot = 0 ;
      for (int spigot=0; spigot<HcalDCCHeader::SPIGOT_COUNT; spigot++) {
          if (!dccHeader->getSpigotPresent(spigot)) continue;

          // Load the given decoder with the pointer and length from this spigot.
          dccHeader->getSpigotData(spigot,htr, fedData.size());
          if ((htr.getFirmwareFlavor()&0xE0)==0x80) continue ; // This is TTP data

          nSpigot++ ;
          // check min length, correct wordcount, empty event, or total length if histo event.
          if ( htr.isUnsuppressed() ) nUS++ ;
          else nZS++ ;
      }

      if ( nUS == nSpigot ) nNZSfed++ ;
      else {
          nZSfed++ ;
          if ( nUS > 0 ) LogWarning("HLTHcalNZSFilter") << "Mixture of ZS(" << nZS
                                                        << ") and NZS(" << nUS
                                                        << ") spigots in FED " << i ;
      }
  }

  if ( (nNZSfed == nFEDs) && (nFEDs > 0) ) { return true ; }
  else {
      if ( nNZSfed > 0 ) LogWarning("HLTHcalNZSFilter") << "Mixture of ZS(" << nZSfed
                                                        << ") and NZS(" << nNZSfed
                                                        << ") FEDs in this event" ;
      return false ;
  }

}

