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
// $Id: HcalDbSource.h,v 1.1 2005/10/04 18:03:03 fedor Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

//
// class decleration
//

class HcalDbSource : 
  public edm::EventSetupRecordIntervalFinder 
{
public:
  HcalDbSource( const edm::ParameterSet& );
  virtual ~HcalDbSource();
  
protected:
  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
                                const edm::IOVSyncValue& iTime, 
                                edm::ValidityInterval& iInterval);
private:
  HcalDbSource (const HcalDbSource&);
  const HcalDbSource& operator= (const HcalDbSource&);
      // ----------member data ---------------------------
};
