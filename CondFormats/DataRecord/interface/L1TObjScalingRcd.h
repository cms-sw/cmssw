#ifndef L1TObjScalingRcd_L1TObjScalingRcd_h
#define L1TObjScalingRcd_L1TObjScalingRcd_h
// -*- C++ -*-
//
// Package:     HLTrigger/HLTcore
// Class  :     L1TObjScalingRcd
//
/**\class L1TObjScalingRcd L1TObjScalingRcd.h HLTrigger/HLTcore/interface/L1TObjScalingRcd.h

 Description: Holds scaling constants for L1T objects.

 Usage:
     Holds constants {A,B,C} such that the pt or Et of a L1T object
     can be scaled by ptScaled = A + B*pt + C*pt^2.
     A cut can then be applied on ptScaled.
     Initial use case is the implementation of the "offline thresholds" 
     in L1T Phase2 in CMS-TDR-021 described in CMS-TDR-021.

*/
//
// Author:      Thiago Tomei
// Created:     Fri, 04 Sep 2020 17:04:33 GMT
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class L1TObjScalingRcd : public edm::eventsetup::EventSetupRecordImplementation<L1TObjScalingRcd> {};

#endif
