#ifndef CondFormatsModuleRcd_HGCalMappingModuleRcd_h
#define CondFormatsModuleRcd_HGCalMappingModuleRcd_h
// -*- C++ -*-
//
// Package:     CondFormats/DataRecords
// Class  :     HGCalMappingModuleRcd
//
/**\class HGCalMappingModuleRcd HGCalMappingModuleRcd.h CondFormats/DataRecords/interface/HGCalMappingModuleRcd.h

 Description: Record for storing mapping parameters used for Electronics Mapping of HGCal Modules

*/
//
// Author:      Pedro Vieira De Castro Ferreira Da Silva
// Created:     Mon, 13 Nov 2023 12:02:11 GMT
//

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Utilities/interface/mplVector.h"
#include "CondFormats/DataRecord/interface/HGCalMappingModuleIndexerRcd.h"

class HGCalMappingModuleRcd
    : public edm::eventsetup::DependentRecordImplementation<HGCalMappingModuleRcd,
                                                            edm::mpl::Vector<HGCalMappingModuleIndexerRcd> > {};

#endif
