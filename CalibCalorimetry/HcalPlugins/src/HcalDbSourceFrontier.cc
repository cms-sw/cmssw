// -*- C++ -*-
//
// Package:    HcalDbSourceFrontier
// Class:      HcalDbSourceFrontier
// 


//
// Original Author:  Fedor Ratnikov
//         Created:  Aug 16, 2005
// $Id$
//
//


#include <iostream>

#include "FWCore/Framework/interface/ESHandle.h"

#include "HcalDbSourceFrontier.h"


 HcalDbSourceFrontier::HcalDbSourceFrontier( const edm::ParameterSet& iConfig )
 {
   std::cout << "HcalDbSourceFrontier::HcalDbSourceFrontier ..." << std::endl;
   this->findingRecord <HcalDbRecord> ();
   setWhatProduced(this);
 }


 HcalDbSourceFrontier::~HcalDbSourceFrontier() {}

// ------------ method called to produce the data  ------------
 HcalDbSourceFrontier::ReturnType
 HcalDbSourceFrontier::produce( const HcalDbRecord& iRecord )
 {

   std::cout << "HcalDbSourceFrontier::produce ..." << std::endl;
  
   std::auto_ptr<HcalDbServiceFrontier> pHcalDbServiceFrontier (new HcalDbServiceFrontier);

    return pHcalDbServiceFrontier ;
 }

 void 
 HcalDbSourceFrontier::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
 				      const edm::IOVSyncValue& iTime, 
 				      edm::ValidityInterval& iInterval) {
   // valid forever for now
   iInterval = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime());
 }


//define this as a plug-in
// DEFINE_FWK_EVENTSETUP_SOURCE(HcalDbSourceFrontier)
  
