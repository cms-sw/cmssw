#ifndef DataRecord_LHCInfoPerLSRcd_h
#define DataRecord_LHCInfoPerLSRcd_h
// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     LHCInfoPerLSRcd
//
/**\class LHCInfoPerLSRcd LHCInfoPerLSRcd.h CondFormats/DataRecord/interface/LHCInfoPerLSRcd.h

 Description: a class containing beam-related parameters and the state of the LHC 
              meant to be stored for every lumisection and to be used instead of LHCInfo


*/

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class LHCInfoPerLSRcd : public edm::eventsetup::EventSetupRecordImplementation<LHCInfoPerLSRcd> {};

#endif
