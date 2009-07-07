// $Id: $
#ifndef LOGICIMP_H 
#define LOGICIMP_H 1

// Include files
#include "L1Trigger/RPCTechnicalTrigger/src/RBCTestLogic.h"
#include "L1Trigger/RPCTechnicalTrigger/src/RBCChamberORLogic.h"
#include "L1Trigger/RPCTechnicalTrigger/src/RBCPatternLogic.h"
#include "L1Trigger/RPCTechnicalTrigger/src/TTUTrackingAlg.h"

RBCTestLogic * createTestLogic();
RBCChamberORLogic * createChamberORLogic();
RBCPatternLogic * createPatternLogic();
TTUTrackingAlg * createTrackingAlg();


#endif // LOGICIMP_H
