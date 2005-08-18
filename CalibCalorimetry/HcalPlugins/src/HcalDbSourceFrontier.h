// -*- C++ -*-
//
// Package:    HcalDbSourceFrontier
// Class:      HcalDbSourceFrontier
// 
/**\class HcalDbSourceFrontier HcalDbSourceFrontier.h 

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Fedor Ratnikov
//         Created:  Wed Aug 16 15:40:06 CDT 2005
// $Id$
//
//

#ifndef HcalDbSourceFrontier_h
#define HcalDbSourceFrontier_h

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbServiceFrontier.h"



//
// class decleration
//

class HcalDbSourceFrontier : 
  public edm::eventsetup::EventSetupRecordIntervalFinder, 
  public edm::eventsetup::ESProducer 
{
public:
  HcalDbSourceFrontier( const edm::ParameterSet& );
  ~HcalDbSourceFrontier();
  
  typedef std::auto_ptr<HcalDbServiceFrontier> ReturnType;
  
  ReturnType produce( const HcalDbRecord& );

protected:
   virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
                                const edm::IOVSyncValue& iTime, 
                                edm::ValidityInterval& iInterval);
private:
  HcalDbSourceFrontier (const HcalDbSourceFrontier&);
  const HcalDbSourceFrontier& operator= (const HcalDbSourceFrontier&);
      // ----------member data ---------------------------
};

#endif
