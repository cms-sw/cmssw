#ifndef DataRecord_HcalRespCorrsRcd_h
#define DataRecord_HcalRespCorrsRcd_h
// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     HcalRespCorrsRcd
// 
/**\class HcalRespCorrsRcd HcalRespCorrsRcd.h CondFormats/DataRecord/interface/HcalRespCorrsRcd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      
// Created:     Sat Mar  1 15:49:28 CET 2008
// $Id: HcalRespCorrsRcd.h,v 1.2 2012/11/12 21:13:54 dlange Exp $
//

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HcalRespCorrsRcd : public edm::eventsetup::DependentRecordImplementation<HcalRespCorrsRcd, boost::mpl::vector<IdealGeometryRecord> > {};

#endif
