/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2002                                                      *
*                                                                              *
*******************************************************************************/
#include "L1Trigger/RPCTrigger/src/L1RpcPattern.h"

L1RpcPattern::L1RpcPattern() {
  Number = -1; //empty pattern
  Code = 0;    
  Sign = 0;

  for(int logPlane = RPCParam::FIRST_PLANE; logPlane <= RPCParam::LAST_PLANE; logPlane++) {
    SetStripFrom(logPlane, RPCParam::NOT_CONECTED);
    SetStripTo(logPlane, RPCParam::NOT_CONECTED + 1);
  }
  //other parameters unset
}

