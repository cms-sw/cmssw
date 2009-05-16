// $Id: LogicImp.cc,v 1.1 2009/01/30 15:42:47 aosorio Exp $
// Include files 

// local
#include "L1Trigger/RPCTechnicalTrigger/interface/LogicImp.h"

//-----------------------------------------------------------------------------
// Logic Factory: Implementation
//
// 2008-10-12 : Andres Osorio
//-----------------------------------------------------------------------------

RBCTestLogic * createTestLogic() { return new RBCTestLogic() ;}
RBCChamberORLogic * createChamberORLogic() { return new RBCChamberORLogic() ;}
RBCPatternLogic * createPatternLogic() { return new RBCPatternLogic() ;}
TTUTrackingAlg * createTrackingAlg() { return new TTUTrackingAlg() ;}

