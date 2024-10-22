#ifndef DataRecord_L1TriggerKeyRcd_h
#define DataRecord_L1TriggerKeyRcd_h
// -*- C++ -*-
//
// Package:     DataRecord
// Class  :     L1TriggerKeyRcd
//
/**\class L1TriggerKeyRcd L1TriggerKeyRcd.h CondFormats/DataRecord/interface/L1TriggerKeyRcd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Giedrius Bacevicius
// Created:     Tue Jul 17 19:15:08 CEST 2007
// $Id: L1TriggerKeyRcd.h,v 1.1 2007/08/22 14:20:13 jbrooke Exp $
//

#include "FWCore/Utilities/interface/mplVector.h"

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"

//class L1TriggerKeyRcd : public edm::eventsetup::EventSetupRecordImplementation<L1TriggerKeyRcd> {};

class L1TriggerKeyRcd
    : public edm::eventsetup::DependentRecordImplementation<L1TriggerKeyRcd, edm::mpl::Vector<L1TriggerKeyListRcd> > {};

#endif
