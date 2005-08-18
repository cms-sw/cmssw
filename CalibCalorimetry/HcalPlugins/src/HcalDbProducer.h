// -*- C++ -*-
//
// Package:    HcalDbProducer
// Class:      HcalDbProducer
// 
/**\class HcalDbProducer HcalDbProducer.h CalibFormats/HcalDbProducer/interface/HcalDbProducer.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Fedor Ratnikov
//         Created:  Tue Aug  9 19:10:10 CDT 2005
// $Id$
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "CalibFormats/HcalObjects/interface/HcalDbServiceHandle.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

class HcalDbProducer : public edm::eventsetup::ESProducer {
   public:
      HcalDbProducer( const edm::ParameterSet& );
      ~HcalDbProducer();

      typedef std::auto_ptr<HcalDbServiceHandle> ReturnType;

      ReturnType produce( const HcalDbRecord& );
   private:
      // ----------member data ---------------------------
  std::string mDbSourceName;
};
