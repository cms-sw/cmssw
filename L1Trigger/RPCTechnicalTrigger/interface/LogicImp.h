// $Id: LogicImp.h,v 1.1 2009/05/16 19:43:30 aosorio Exp $
#ifndef LOGICIMP_H 
#define LOGICIMP_H 1

// Include files
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCTestLogic.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCChamberORLogic.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCPatternLogic.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUTrackingAlg.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUSectorORLogic.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUTwoORLogic.h"


RBCTestLogic      * createTestLogic();
RBCChamberORLogic * createChamberORLogic();
RBCPatternLogic   * createPatternLogic();
TTUTrackingAlg    * createTrackingAlg();
TTUSectorORLogic  * createSectorORLogic();
TTUTwoORLogic     * createTwoORLogic();

#endif // LOGICIMP_H
