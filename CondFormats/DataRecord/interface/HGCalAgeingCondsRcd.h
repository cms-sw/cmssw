#ifndef CondFormats_HGCalAgeingCondsRcd_h
#define CondFormats_HGCalAgeingCondsRcd_h
// -*- C++ -*-
//
// Package:     CondFormats/DataRecord
// Class  :     HGCalAgeingCondsRcd
//
/**\class HGCalAgeingCondsRcd HGCalAgeingCondsRcd.h CondFormats/DataRecord/interface/HGCalAgeingCondsRcd.h
 *
 * Description:
 *   This record is used to generate the ageing conditions for the SIM-DIGI step based on the information collected by the HGCalDenseIndexer
 *   This record depends on the HGCalDenseIndexInfoRcd
 *
 */
//
// Author:      Pedro Silva
// Created:     Fri, 02 Oct 2026 18:18:02 GMT
//

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Utilities/interface/mplVector.h"
#include "CondFormats/DataRecord/interface/HGCalElectronicsMappingRcd.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "CondFormats/DataRecord/interface/HGCalDenseIndexInfoRcd.h"

class HGCalAgeingCondsRcd
    : public edm::eventsetup::DependentRecordImplementation<
          HGCalAgeingCondsRcd,
          edm::mpl::Vector<HGCalElectronicsMappingRcd, CaloGeometryRecord, HGCalDenseIndexInfoRcd> > {};

#endif
