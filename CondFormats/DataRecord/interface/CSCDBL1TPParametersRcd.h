#ifndef DataRecord_CSCDBL1TPParametersRcd_h

#define DataRecord_CSCDBL1TPParametersRcd_h



// -*- C++ -*-

//

// Package:     DataRecord

// Class  :     CSCDBL1TPParametersRcd

// 

/** \class CSCDBL1TPParametersRcd CSCDBL1TPParametersRcd.h CondFormats/DataRecord/interface/CSCDBL1TPParametersRcd.h

*

* Description: Record for configuration parameters needed for the Level-1 CSC

*              Trigger Primitives emulator.

*/

//

// Author:      Slava Valuev

// Created:     Thu Apr 12 11:18:05 CEST 2007

// $Id: CSCDBL1TPParametersRcd.h,v 1.1 2008/07/06 05:00:29 wsun Exp $

//

#include "boost/mpl/vector.hpp"

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"



//class CSCDBL1TPParametersRcd : public edm::eventsetup::EventSetupRecordImplementation<CSCDBL1TPParametersRcd> {};
class CSCDBL1TPParametersRcd : public edm::eventsetup::DependentRecordImplementation<CSCDBL1TPParametersRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};



#endif
