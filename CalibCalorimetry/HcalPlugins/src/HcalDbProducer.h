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
// $Id: HcalDbProducer.h,v 1.2 2005/10/04 18:03:03 fedor Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

class HcalDbService;
class HcalDbRecord;
class HcalPedestals;
class HcalPedestalWidths;
class HcalGains;
class HcalGainWidths;
class HcalPedestalsRcd;
class HcalPedestalWidthsRcd;
class HcalGainsRcd;
class HcalGainWidthsRcd;

class HcalDbProducer : public edm::ESProducer {
 public:
  HcalDbProducer( const edm::ParameterSet& );
  ~HcalDbProducer();
  
  typedef std::auto_ptr<HcalDbService> ReturnType;
  
  ReturnType produce( const HcalDbRecord& );

  // callbacks
  void poolPedestalsCallback (const HcalPedestalsRcd& fRecord);
  void poolPedestalWidthsCallback (const HcalPedestalWidthsRcd& fRecord);
  void poolGainsCallback (const HcalGainsRcd& fRecord);
  void poolGainWidthsCallback (const HcalGainWidthsRcd& fRecord);

   private:
      // ----------member data ---------------------------
  std::string mDbSourceName;
  const HcalPedestals* mPedestals;
  const HcalPedestalWidths* mPedestalWidths;
  const HcalGains* mGains;
  const HcalGainWidths* mGainWidths;
};
