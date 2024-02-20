#ifndef CondFormatsModuleRcd_HGCalElectronicsMappingRcd_h
#define CondFormatsModuleRcd_HGCalElectronicsMappingRcd_h
// -*- C++ -*-
//
// Package:     CondFormats/DataRecords
// Class  :     HGCalElectronicsMappingRcd
//
/**\class HGCalElectronicsMappingRcd HGCalElectronicsMappingRcd.h CondFormats/DataRecords/interface/HGCalElectronicsMappingRcd.h

 Description: This record is used to store the electronics mapping parameters for HGCAL
 It contains the readout indexer classes for modules and cells as well as the characteristics of modules and cells stored in SoA

*/
//
// Author:      Pedro Vieira De Castro Ferreira Da Silva
// Created:     Mon, 13 Nov 2023 12:02:11 GMT
//

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Utilities/interface/mplVector.h"

class HGCalElectronicsMappingRcd : public edm::eventsetup::EventSetupRecordImplementation<HGCalElectronicsMappingRcd> {
};


#endif
