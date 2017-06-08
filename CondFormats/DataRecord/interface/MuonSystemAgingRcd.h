#ifndef MuonSystemAgingRcd_MuonSystemAgingRcd_h
#define MuonSystemAgingRcd_MuonSystemAgingRcd_h
// -*- C++ -*-
//
// Package:     CondFormats/DataRecord
// Class  :     MuonSystemAgingRcd
// 
/**\class MuonSystemAgingRcd MuonSystemAgingRcd.h CondFormats/DataRecord/interface/MuonSystemAgingRcd.h

 Description: CondFormat implementing inefficiencies in muon stations to mimic aging effects 

*/
//
// Author:      Sunil Bansal
// Created:     Wed, 01 Jun 2016 13:03:56 GMT
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class MuonSystemAgingRcd : public edm::eventsetup::EventSetupRecordImplementation<MuonSystemAgingRcd> {};

#endif
