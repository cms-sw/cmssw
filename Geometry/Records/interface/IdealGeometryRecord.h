#ifndef RECORDS_IDEALGEOMETRYRECORD_H
#define RECORDS_IDEALGEOMETRYRECORD_H
// -*- C++ -*-
//
// Package:     Records
// Class  :     IdealGeometryRecord
//
/**\class IdealGeometryRecord IdealGeometryRecord.h Geometry/Records/interface/IdealGeometryRecord.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:
// Created:     Mon Jul 25 11:05:09 EDT 2005
//

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/GeometryFileRcd.h"
#include "Geometry/Records/interface/PGeometricTimingDetExtraRcd.h"
#include "FWCore/Utilities/interface/mplVector.h"

class IdealGeometryRecord : public edm::eventsetup::DependentRecordImplementation<
                                IdealGeometryRecord,
                                edm::mpl::Vector<GeometryFileRcd, PGeometricTimingDetExtraRcd> > {};

#endif /* RECORDS_IDEALGEOMETRYRECORD_H */
