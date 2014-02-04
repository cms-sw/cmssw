#ifndef DataRecord_HcalCovarianceMatricesRcd_h
#define DataRecord_HcalCovarianceMatricesRcd_h
// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     HcalCovarianceMatricesRcd
// 
/**\class HcalCovarianceMatricesRcd HcalCovarianceMatricesRcd.h CondFormats/DataRecord/interface/HcalCovarianceMatricesRcd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      
// Created:     Sat Mar  1 15:49:28 CET 2008
// $Id: HcalCovarianceMatricesRcd.h,v 1.2 2012/11/12 21:13:54 dlange Exp $
//

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

class HcalCovarianceMatricesRcd : public edm::eventsetup::DependentRecordImplementation<HcalCovarianceMatricesRcd, boost::mpl::vector<HcalRecNumberingRecord> > {};

#endif
