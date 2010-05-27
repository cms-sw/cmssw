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
// $Id: IdealGeometryRecord.h,v 1.3 2009/03/26 08:17:06 fambrogl Exp $
//

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/GeometryFileRcd.h"
#include "boost/mpl/vector.hpp"

class IdealGeometryRecord : public edm::eventsetup::DependentRecordImplementation<
  IdealGeometryRecord, boost::mpl::vector<GeometryFileRcd> > { };

#endif /* RECORDS_IDEALGEOMETRYRECORD_H */

