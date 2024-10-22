#ifndef LHCInfoPerFillRcd_LHCInfoPerFillRcd_h
#define LHCInfoPerFillRcd_LHCInfoPerFillRcd_h
// -*- C++ -*-
//
// Package:     CondFormats/DataRecord
// Class  :     LHCInfoPerFillRcd
//
/**\class LHCInfoPerFillRcd LHCInfoPerFillRcd.h CondFormats/DataRecord/interface/LHCInfoPerFillRcd.h

 Description: a class containing beam-related parameters and the state of the LHC 
              meant to be stored once or twice per fill and to be used instead of LHCInfo

 Usage:
    <usage>

*/

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class LHCInfoPerFillRcd : public edm::eventsetup::EventSetupRecordImplementation<LHCInfoPerFillRcd> {};

#endif
