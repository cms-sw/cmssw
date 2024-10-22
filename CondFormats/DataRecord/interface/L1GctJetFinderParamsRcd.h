#ifndef DataRecord_L1GctJetFinderParamsRcd_h
#define DataRecord_L1GctJetFinderParamsRcd_h
// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     L1GctJetFinderParamsRcd
//
/**\class L1GctJetFinderParamsRcd L1GctJetFinderParamsRcd.h CondFormats/DataRecord/interface/L1GctJetFinderParamsRcd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:
// Created:     Tue Jul 10 10:14:03 CEST 2007
// $Id: L1GctJetFinderParamsRcd.h,v 1.2 2008/03/03 07:09:47 wsun Exp $
//

#include "FWCore/Utilities/interface/mplVector.h"

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"
#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"

//class L1GctJetFinderParamsRcd : public edm::eventsetup::EventSetupRecordImplementation<L1GctJetFinderParamsRcd> {};
class L1GctJetFinderParamsRcd : public edm::eventsetup::DependentRecordImplementation<
                                    L1GctJetFinderParamsRcd,
                                    edm::mpl::Vector<L1TriggerKeyListRcd, L1TriggerKeyRcd, L1CaloGeometryRecord> > {};

#endif
