// $Id: LogicImp.h,v 1.2 2009/06/17 15:27:23 aosorio Exp $
#ifndef LOGICIMP_H 
#define LOGICIMP_H 1

// Include files
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCTestLogic.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCChamberORLogic.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCPatternLogic.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUTrackingAlg.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUSectorORLogic.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUTwoORLogic.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUWedgeORLogic.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUPointingLogic.h"


RBCTestLogic      * createTestLogic();
RBCChamberORLogic * createChamberORLogic();
RBCPatternLogic   * createPatternLogic();
TTUTrackingAlg    * createTrackingAlg();
TTUSectorORLogic  * createSectorORLogic();
TTUTwoORLogic     * createTwoORLogic();
TTUWedgeORLogic   * createWedgeORLogic();
TTUPointingLogic  * createPointingLogic();

#endif // LOGICIMP_H
