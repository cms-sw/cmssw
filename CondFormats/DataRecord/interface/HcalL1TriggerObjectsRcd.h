#ifndef DataRecord_HcalL1TriggerObjectsRcd_h
#define DataRecord_HcalL1TriggerObjectsRcd_h
// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     HcalL1TriggerObjectsRcd
// 
/**\class HcalL1TriggerObjectsRcd HcalL1TriggerObjectsRcd.h CondFormats/DataRecord/interface/HcalL1TriggerObjectsRcd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      
// Created:     Fri Nov  7 18:37:16 CET 2008
// $Id: HcalL1TriggerObjectsRcd.h,v 1.1 2008/11/08 21:19:31 rofierzy Exp $
//

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HcalL1TriggerObjectsRcd : public edm::eventsetup::DependentRecordImplementation<HcalL1TriggerObjectsRcd, boost::mpl::vector<IdealGeometryRecord> > {};

#endif
