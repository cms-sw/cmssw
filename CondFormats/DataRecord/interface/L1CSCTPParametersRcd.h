#ifndef DataRecord_L1CSCTPParametersRcd_h

#define DataRecord_L1CSCTPParametersRcd_h



// -*- C++ -*-

//

// Package:     DataRecord

// Class  :     L1CSCTPParametersRcd

// 

/** \class L1CSCTPParametersRcd L1CSCTPParametersRcd.h CondFormats/DataRecord/interface/L1CSCTPParametersRcd.h

*

* Description: Record for configuration parameters needed for the Level-1 CSC

*              Trigger Primitives emulator.

*/

//

// Author:      Slava Valuev

// Created:     Thu Apr 12 11:18:05 CEST 2007

// $Id: L1CSCTPParametersRcd.h,v 1.1 2007/04/19 15:52:08 jbrooke Exp $

//

#include "boost/mpl/vector.hpp"

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"



//class L1CSCTPParametersRcd : public edm::eventsetup::EventSetupRecordImplementation<L1CSCTPParametersRcd> {};
class L1CSCTPParametersRcd : public edm::eventsetup::DependentRecordImplementation<L1CSCTPParametersRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};



#endif
