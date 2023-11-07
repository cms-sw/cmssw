#ifndef CondFormats_HGCalDenseIndexInfoRcd_h
#define CondFormats_HGCalDenseIndexInfoRcd_h
// -*- C++ -*-
//
// Package:     CondFormats/DataRecord
// Class  :     HGCalDenseIndexInfoRcd
//
/**\class HGCalDenseIndexInfoRcd HGCalDenseIndexInfoRcd.h CondFormats/DataRecord/interface/HGCalDenseIndexInfoRcd.h
 *
 * Description:
 *   This record is used join information from the geometry and logical mapping
 *   This record depends on the HGCalElectronicsMappingRcd and CaloGeometryRecord
 *
 */
//
// Author:      Pedro Da Silva, Izaak Neutelings
// Created:     Mon, 29 May 2023 09:13:07 GMT
//

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Utilities/interface/mplVector.h"
#include "CondFormats/DataRecord/interface/HGCalElectronicsMappingRcd.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

class HGCalDenseIndexInfoRcd : public edm::eventsetup::DependentRecordImplementation<
                                   HGCalDenseIndexInfoRcd,
                                   edm::mpl::Vector<HGCalElectronicsMappingRcd, CaloGeometryRecord> > {};

#endif
