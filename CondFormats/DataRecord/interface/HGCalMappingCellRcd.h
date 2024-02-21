#ifndef CondFormatsModuleRcd_HGCalMappingCellRcd_h
#define CondFormatsModuleRcd_HGCalMappingCellRcd_h
// -*- C++ -*-
//
// Package:     CondFormats/DataRecords
// Class  :     HGCalMappingCellRcd
//
/**\class HGCalMappingCellRcd HGCalMappingCellRcd.h CondFormats/DataRecords/interface/HGCalMappingCellRcd.h

 Description: Record for storing the parameters describing a readout cell within each HGCAL module, used for the electronics mapping

*/
//
// Author:      Pedro Vieira De Castro Ferreira Da Silva
// Created:     Mon, 13 Nov 2023 12:02:11 GMT
//

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Utilities/interface/mplVector.h"
#include "CondFormats/DataRecord/interface/HGCalMappingCellIndexerRcd.h"

class HGCalMappingCellRcd
    : public edm::eventsetup::DependentRecordImplementation<HGCalMappingCellRcd,
                                                            edm::mpl::Vector<HGCalMappingCellIndexerRcd> > {};

#endif
