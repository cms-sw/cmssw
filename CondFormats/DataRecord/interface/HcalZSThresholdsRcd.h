#ifndef DataRecord_HcalZSThresholdsRcd_h
#define DataRecord_HcalZSThresholdsRcd_h
// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     HcalZSThresholdsRcd
// 
/**\class HcalZSThresholdsRcd HcalZSThresholdsRcd.h CondFormats/DataRecord/interface/HcalZSThresholdsRcd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      
// Created:     Sat Nov 24 16:42:00 CET 2007
// $Id: HcalZSThresholdsRcd.h,v 1.2 2012/11/12 21:13:55 dlange Exp $
//

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

class HcalZSThresholdsRcd : public edm::eventsetup::DependentRecordImplementation<HcalZSThresholdsRcd, boost::mpl::vector<HcalRecNumberingRecord> > {};

#endif
