#ifndef DataRecord_HcalCholeskyMatricesRcd_h
#define DataRecord_HcalCholeskyMatricesRcd_h
// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     HcalCholeskyMatricesRcd
// 
/**\class HcalCholeskyMatricesRcd HcalCholeskyMatricesRcd.h CondFormats/DataRecord/interface/HcalCholeskyMatricesRcd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      
// Created:     Sat Mar  1 15:49:28 CET 2008
// $Id: HcalCholeskyMatricesRcd.h,v 1.2 2012/11/12 21:13:54 dlange Exp $
//

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

class HcalCholeskyMatricesRcd : public edm::eventsetup::DependentRecordImplementation<HcalCholeskyMatricesRcd, boost::mpl::vector<HcalRecNumberingRecord> > {};

#endif
