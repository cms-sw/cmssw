#ifndef CondFormats_HGCalModuleConfigurationRcd_h
#define CondFormats_HGCalModuleConfigurationRcd_h
// -*- C++ -*-
//
// Package:     CondFormats/DataRecord
// Class  :     HGCalModuleConfigurationRcd
//
/**\class HGCalModuleConfigurationRcd HGCalModuleConfigurationRcd.h CondFormats/DataRecord/interface/HGCalModuleConfigurationRcd.h
 *
 * Description:
 *   This record is used for passing the configuration parameters to the calibration step in RAW -> RECO,
 *   This record depends on the HGCalMappingModuleIndexerRcd.
 *
 */
//
// Author:      Pedro Da Silva, Izaak Neutelings
// Created:     Mon, 29 May 2023 09:13:07 GMT
//

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Utilities/interface/mplVector.h"
#include "CondFormats/DataRecord/interface/HGCalElectronicsMappingRcd.h"

class HGCalModuleConfigurationRcd
    : public edm::eventsetup::DependentRecordImplementation<HGCalModuleConfigurationRcd,
                                                            edm::mpl::Vector<HGCalElectronicsMappingRcd> > {};

#endif
