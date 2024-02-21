#ifndef CondFormatsModuleRcd_HGCalMappingCellIndexerRcd_h
#define CondFormatsModuleRcd_HGCalMappingCellIndexerRcd_h
// -*- C++ -*-
//
// Package:     CondFormats/DataRecords
// Class  :     HGCalMappingCellIndexerRcd
//
/**\class HGCalMappingCellIndexerRcd HGCalMappingCellIndexerRcd.h CondFormats/DataRecords/interface/HGCalMappingCellIndexerRcd.h

 Description: This record is used to store the cell readout indexer class for the electronics mapping parameters for HGCAL

*/
//
// Author:      Pedro Vieira De Castro Ferreira Da Silva
// Created:     Mon, 13 Nov 2023 12:02:11 GMT
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class HGCalMappingCellIndexerRcd : public edm::eventsetup::EventSetupRecordImplementation<HGCalMappingCellIndexerRcd> {
};

#endif
