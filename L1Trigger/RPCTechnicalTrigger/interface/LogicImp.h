// $Id: LogicImp.h,v 1.1 2009/02/05 13:46:21 aosorio Exp $
#ifndef LOGICIMP_H 
#define LOGICIMP_H 1

// Include files
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCTestLogic.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCChamberORLogic.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCPatternLogic.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUTrackingAlg.h"

RBCTestLogic * createTestLogic();
RBCChamberORLogic * createChamberORLogic();
RBCPatternLogic * createPatternLogic();
TTUTrackingAlg * createTrackingAlg();


#endif // LOGICIMP_H
