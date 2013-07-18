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
// $Id: IdealGeometryRecord.h,v 1.4 2010/05/27 23:14:26 case Exp $
//

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/GeometryFileRcd.h"
#include "Geometry/Records/interface/PGeometricDetExtraRcd.h"
#include "boost/mpl/vector.hpp"

class IdealGeometryRecord : public edm::eventsetup::DependentRecordImplementation<
  IdealGeometryRecord, boost::mpl::vector<GeometryFileRcd, PGeometricDetExtraRcd> > { };

#endif /* RECORDS_IDEALGEOMETRYRECORD_H */

