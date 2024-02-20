#ifndef CondFormatsModuleRcd_HGCalElectronicsMappingRcd_h
#define CondFormatsModuleRcd_HGCalElectronicsMappingRcd_h
// -*- C++ -*-
//
// Package:     CondFormats/DataRecords
// Class  :     HGCalElectronicsMappingRcd
//
/**\class HGCalElectronicsMappingRcd HGCalElectronicsMappingRcd.h CondFormats/DataRecords/interface/HGCalElectronicsMappingRcd.h
<<<<<<< HEAD
 Description: This record *is temporary* and it is used to store the parameters which are used to describe 
 - dense indexing of HGCAL modules and corresponding cells in the readound sequence
 - module and cell information in the electronics mapping
 The record will change to its final format(s) once the we settle on the final formats to be used in CondDB
=======

 Description: This record is used to store the electronics mapping parameters for HGCAL
 It contains the readout indexer classes for modules and cells as well as the characteristics of modules and cells stored in SoA

>>>>>>> f199c5da48a (moving to single record as all conds produced have the same IOV)
*/
//
// Author:      Pedro Vieira De Castro Ferreira Da Silva
// Created:     Mon, 13 Nov 2023 12:02:11 GMT
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class HGCalElectronicsMappingRcd : public edm::eventsetup::EventSetupRecordImplementation<HGCalElectronicsMappingRcd> {
};

#endif
