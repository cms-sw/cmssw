#ifndef CASTORDBPRODUCER_CASTORDBRECORD_H
#define CASTORDBPRODUCER_CASTORDBRECORD_H
// -*- C++ -*-
//
// Package:     CastorDbProducer
// Class  :     CastorDbRecord
//
/**\class CastorDbRecord CastorDbRecord.h CalibFormats/CastorDbProducer/interface/CastorDbRecord.h

 Description: based on the HCAL Db record class

 Usage:
    <usage>
*/

#include "boost/mpl/vector.hpp"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
// #include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

#include "CondFormats/DataRecord/interface/CastorChannelQualityRcd.h"
#include "CondFormats/DataRecord/interface/CastorElectronicsMapRcd.h"
#include "CondFormats/DataRecord/interface/CastorElectronicsMapRcd.h"
#include "CondFormats/DataRecord/interface/CastorGainWidthsRcd.h"
#include "CondFormats/DataRecord/interface/CastorGainsRcd.h"
#include "CondFormats/DataRecord/interface/CastorPedestalWidthsRcd.h"
#include "CondFormats/DataRecord/interface/CastorPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/CastorQIEDataRcd.h"

class CastorDbRecord
    : public edm::eventsetup::DependentRecordImplementation<CastorDbRecord,
                                                            boost::mpl::vector<CastorPedestalsRcd,
                                                                               CastorPedestalWidthsRcd,
                                                                               CastorGainsRcd,
                                                                               CastorGainWidthsRcd,
                                                                               CastorQIEDataRcd,
                                                                               CastorChannelQualityRcd,
                                                                               CastorElectronicsMapRcd> > {};

#endif /* CASTORDBPRODUCER_CASTORDBRECORD_H */
