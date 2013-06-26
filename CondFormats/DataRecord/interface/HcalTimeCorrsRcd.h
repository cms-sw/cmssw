#ifndef DataRecord_HcalTimeCorrsRcd_h
#define DataRecord_HcalTimeCorrsRcd_h
// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     HcalTimeCorrsRcd
// 
/**\class HcalTimeCorrsRcd HcalTimeCorrsRcd.h CondFormats/DataRecord/interface/HcalTimeCorrsRcd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      
// Created:     Sat Mar  1 15:49:28 CET 2008
// $Id: HcalTimeCorrsRcd.h,v 1.2 2012/11/12 21:13:55 dlange Exp $
//

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HcalTimeCorrsRcd : public edm::eventsetup::DependentRecordImplementation<HcalTimeCorrsRcd, boost::mpl::vector<IdealGeometryRecord> > {};

#endif
