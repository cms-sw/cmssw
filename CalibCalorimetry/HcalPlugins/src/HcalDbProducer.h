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
// $Id: HcalDbProducer.h,v 1.8 2005/12/15 23:37:58 fedor Exp $
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

#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "CondFormats/DataRecord/interface/HcalElectronicsMapRcd.h"  	 
#include "CondFormats/DataRecord/interface/HcalGainWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalGainsRcd.h"
#include "CondFormats/DataRecord/interface/HcalPedestalWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalPedestalsRcd.h" 
#include "CondFormats/DataRecord/interface/HcalQIEDataRcd.h"




class HcalDbProducer : public edm::ESProducer {
 public:
  HcalDbProducer( const edm::ParameterSet& );
  ~HcalDbProducer();
  
  boost::shared_ptr<HcalDbService> produce( const HcalDbRecord& );

  // callbacks
  void pedestalsCallback (const HcalPedestalsRcd& fRecord);
  void pedestalWidthsCallback (const HcalPedestalWidthsRcd& fRecord);
  void gainsCallback (const HcalGainsRcd& fRecord);
  void gainWidthsCallback (const HcalGainWidthsRcd& fRecord);
  void QIEDataCallback (const HcalQIEDataRcd& fRecord);
  void channelQualityCallback (const HcalChannelQualityRcd& fRecord);
  void electronicsMapCallback (const HcalElectronicsMapRcd& fRecord);

   private:
      // ----------member data ---------------------------
  boost::shared_ptr<HcalDbService> mService;
  std::vector<std::string> mDumpRequest;
  std::ostream* mDumpStream;
};
