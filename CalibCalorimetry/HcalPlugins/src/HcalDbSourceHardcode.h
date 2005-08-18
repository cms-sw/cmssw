// -*- C++ -*-
//
// Package:    HcalDbSourceHardcode
// Class:      HcalDbSourceHardcode
// 
/**\class HcalDbSourceHardcode HcalDbSourceHardcode.h CalibFormats/HcalDbSourceHardcode/interface/HcalDbSourceHardcode.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Fedor Ratnikov
//         Created:  Wed Aug 10 15:40:06 CDT 2005
// $Id$
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
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbServiceHardcode.h"



//
// class decleration
//

class HcalDbSourceHardcode : 
  public edm::eventsetup::EventSetupRecordIntervalFinder, 
  public edm::eventsetup::ESProducer 
{
public:
  HcalDbSourceHardcode( const edm::ParameterSet& );
  ~HcalDbSourceHardcode();
  
  typedef std::auto_ptr<HcalDbServiceHardcode> ReturnType;
  
  ReturnType produce( const HcalDbRecord& );

protected:
   virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
                                const edm::IOVSyncValue& iTime, 
                                edm::ValidityInterval& iInterval);
private:
  HcalDbSourceHardcode (const HcalDbSourceHardcode&);
  const HcalDbSourceHardcode& operator= (const HcalDbSourceHardcode&);
      // ----------member data ---------------------------
};
