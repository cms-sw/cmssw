#ifndef GeometryRecord_HcalRecNumberingRecord_h
#define GeometryRecord_HcalRecNumberingRecord_h
// -*- C++ -*-
//
// Package:     Record
// Class  :     HcalRecNumberingRecord
// 
/**\class HcalRecNumberingRecord HcalRecNumberingRecord.h Geometry/Record/interface/HcalRecNumberingRecord.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      
// Created:     Thu Dec 24 16:41:02 PDT 2013
// $Id: HcalRecNumberingRecord.h,v 1.0 2013/12/25 10:22:50 sunanda Exp $
//

#include <boost/mpl/vector.hpp>
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/HcalSimNumberingRecord.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

class HcalRecNumberingRecord : public edm::eventsetup::DependentRecordImplementation<HcalRecNumberingRecord, boost::mpl::vector<IdealGeometryRecord, HcalSimNumberingRecord> > {};

#endif
