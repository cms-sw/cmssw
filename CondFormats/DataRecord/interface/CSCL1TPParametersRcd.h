#ifndef DataRecord_CSCL1TPParametersRcd_h

#define DataRecord_CSCL1TPParametersRcd_h



// -*- C++ -*-

//

// Package:     DataRecord

// Class  :     CSCL1TPParametersRcd

// 

/** \class CSCL1TPParametersRcd CSCL1TPParametersRcd.h CondFormats/DataRecord/interface/CSCL1TPParametersRcd.h

*

* Description: Record for configuration parameters needed for the Level-1 CSC

*              Trigger Primitives emulator.

*/

//

// Author:      Slava Valuev

// Created:     Thu Apr 12 11:18:05 CEST 2007

// $Id: CSCL1TPParametersRcd.h,v 1.2 2008/03/03 07:09:47 wsun Exp $

//

#include "boost/mpl/vector.hpp"

//#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"



//class CSCL1TPParametersRcd : public edm::eventsetup::EventSetupRecordImplementation<CSCL1TPParametersRcd> {};
class CSCL1TPParametersRcd : public edm::eventsetup::DependentRecordImplementation<CSCL1TPParametersRcd, boost::mpl::vector<L1TriggerKeyListRcd,L1TriggerKeyRcd> > {};



#endif
