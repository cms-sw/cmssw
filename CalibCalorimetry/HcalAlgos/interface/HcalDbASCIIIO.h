//
// F.Ratnikov (UMd), Jul. 19, 2005
//
#ifndef HcalDbASCIIIO_h
#define HcalDbASCIIIO_h

#include <iostream>

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CondFormats/HcalObjects/interface/AllClasses.h"

/**
   \class HcalDbASCIIIO
   \brief IO for ASCII instances of Hcal Calibrations
   \author Fedor Ratnikov Oct. 28, 2005
   $Id: HcalDbProducer.h,v 1.2 2005/10/04 18:03:03 fedor Exp $
   
*/
namespace HcalDbASCIIIO {
  bool getObject (std::istream& fInput, HcalPedestals* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalPedestals& fObject);
  bool getObject (std::istream& fInput, HcalPedestalWidths* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalPedestalWidths& fObject);
  bool getObject (std::istream& fInput, HcalGains* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalGains& fObject);
  bool getObject (std::istream& fInput, HcalGainWidths* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalGainWidths& fObject);
  bool getObject (std::istream& fInput, HcalQIEShape* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalQIEShape& fObject);
  bool getObject (std::istream& fInput, HcalQIEData* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalQIEData& fObject);
  bool getObject (std::istream& fInput, HcalChannelQuality* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalChannelQuality& fObject);
  bool getObject (std::istream& fInput, HcalElectronicsMap* fObject);
  bool dumpObject (std::ostream& fOutput, const HcalElectronicsMap& fObject);
} 
#endif
