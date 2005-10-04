// -*- C++ -*-
//
// Package:    HcalDbSource
// Class:      HcalDbSource
// 
/**\class HcalDbSource HcalDbSource.h CalibFormats/HcalDbSource/interface/HcalDbSource.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Fedor Ratnikov
//         Created:  Wed Aug 10 15:40:06 CDT 2005
// $Id: HcalDbSource.cc,v 1.2 2005/08/19 17:20:48 fedor Exp $
//
//


#include <iostream>

#include "FWCore/Framework/interface/ESHandle.h"

#include "HcalDbSource.h"


 HcalDbSource::HcalDbSource( const edm::ParameterSet& iConfig )
 {
   std::cout << "HcalDbSource::HcalDbSource ..." << std::endl;
   findingRecord <HcalDbRecord> ();
 }


 HcalDbSource::~HcalDbSource() {}

// ------------ method called to produce the data  ------------

 void 
 HcalDbSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
 				      const edm::IOVSyncValue& iTime, 
 				      edm::ValidityInterval& iInterval) {
   std::cout << "HcalDbSource::setIntervalFor-> Current run is " << iTime.eventID() << std::endl;
   //Be valid for 3 time steps
   edm::EventID newTime = edm::EventID( (iTime.eventID().run() - 1 ) - ((iTime.eventID().run() - 1 ) %3) +1);
   edm::EventID endTime = newTime.nextRun().nextRun().nextRun().previousRunLastEvent();
   iInterval = edm::ValidityInterval( edm::IOVSyncValue( newTime),
				      edm::IOVSyncValue(endTime) );
   std::cout << "HcalDbSource::setIntervalFor-> new interval " << newTime << "->" <<  endTime << std::endl;
 }

//define this as a plug-in
// DEFINE_FWK_EVENTSETUP_SOURCE(HcalDbSource)
  
