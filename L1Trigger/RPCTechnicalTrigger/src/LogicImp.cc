// $Id: $
// Include files 

// local
#include "L1Trigger/RPCTechnicalTrigger/src/LogicImp.h"

//-----------------------------------------------------------------------------
// Logic Factory: Implementation
//
// 2008-10-12 : Andres Osorio
//-----------------------------------------------------------------------------

RBCTestLogic * createTestLogic() { return new RBCTestLogic() ;}
RBCChamberORLogic * createChamberORLogic() { return new RBCChamberORLogic() ;}
RBCPatternLogic * createPatternLogic() { return new RBCPatternLogic() ;}
TTUTrackingAlg * createTrackingAlg() { return new TTUTrackingAlg() ;}

