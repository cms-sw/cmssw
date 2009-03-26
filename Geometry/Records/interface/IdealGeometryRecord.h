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
// $Id: IdealGeometryRecord.h,v 1.2 2009/02/02 03:48:29 case Exp $
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
//#include "FWCore/Framework/interface/DependentRecordImplementation.h"
//#include "Geometry/Records/interface/GeometryFileRcd.h"
//#include "boost/mpl/vector.hpp"

class IdealGeometryRecord : public edm::eventsetup::EventSetupRecordImplementation<IdealGeometryRecord> {};

//class IdealGeometryRecord : public edm::eventsetup::DependentRecordImplementation<
//  IdealGeometryRecord, boost::mpl::vector<GeometryFileRcd> > { };

#endif /* RECORDS_IDEALGEOMETRYRECORD_H */

