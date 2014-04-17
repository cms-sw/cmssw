#ifndef GeometryRecord_ShashlikNumberingRecord_h
#define GeometryRecord_ShashlikNumberingRecord_h
// -*- C++ -*-
//
// Package:     Record
// Class  :     ShashlikNumberingRecord
// 
/**\class ShashlikNumberingRecord ShashlikNumberingRecord.h Geometry/Record/interface/ShashlikNumberingRecord.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      
// Created:     Thu Mar 21 16:41:02 PDT 2014
// $Id: ShashlikNumberingRecord.h,v 1.0 2014/03/21 10:22:50 sunanda Exp $
//

#include <boost/mpl/vector.hpp>
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

class ShashlikNumberingRecord : public edm::eventsetup::DependentRecordImplementation<ShashlikNumberingRecord, boost::mpl::vector<IdealGeometryRecord> > {};

#endif
