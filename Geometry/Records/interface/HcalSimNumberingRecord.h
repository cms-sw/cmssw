#ifndef GeometryRecord_HcalSimNumberingRecord_h
#define GeometryRecord_HcalSimNumberingRecord_h
// -*- C++ -*-
//
// Package:     Record
// Class  :     HcalSimNumberingRecord
// 
/**\class HcalSimNumberingRecord HcalSimNumberingRecord.h Geometry/Record/interface/HcalSimNumberingRecord.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      
// Created:     Thu Dec 24 16:41:02 PDT 2013
// $Id: HcalSimNumberingRecord.h,v 1.0 2013/12/25 10:22:50 sunanda Exp $
//

#include <boost/mpl/vector.hpp>
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

class HcalSimNumberingRecord : public edm::eventsetup::DependentRecordImplementation<HcalSimNumberingRecord, boost::mpl::vector<IdealGeometryRecord> > {};

#endif
