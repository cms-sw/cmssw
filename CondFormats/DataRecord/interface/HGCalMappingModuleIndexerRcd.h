#ifndef CondFormatsModuleRcd_HGCalMappingModuleIndexerRcd_h
#define CondFormatsModuleRcd_HGCalMappingModuleIndexerRcd_h
// -*- C++ -*-
//
// Package:     CondFormats/DataRecords
// Class  :     HGCalMappingModuleIndexerRcd
//
/**\class HGCalMappingModuleIndexerRcd HGCalMappingModuleIndexerRcd.h CondFormats/DataRecords/interface/HGCalMappingModuleIndexerRcd.h

 Description: This record is used to store the readout indexer class of the modules in the electronics mapping of HGCAL

*/
//
// Author:      Pedro Vieira De Castro Ferreira Da Silva
// Created:     Mon, 13 Nov 2023 12:02:11 GMT
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class HGCalMappingModuleIndexerRcd
    : public edm::eventsetup::EventSetupRecordImplementation<HGCalMappingModuleIndexerRcd> {};

#endif
