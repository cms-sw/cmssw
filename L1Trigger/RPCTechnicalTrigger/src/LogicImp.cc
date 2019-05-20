// Include files

// local
#include "L1Trigger/RPCTechnicalTrigger/interface/LogicImp.h"

//-----------------------------------------------------------------------------
// Logic Factory: Implementation
//
// 2008-10-12 : Andres Osorio
//-----------------------------------------------------------------------------

RBCTestLogic* createTestLogic() { return new RBCTestLogic(); }
RBCChamberORLogic* createChamberORLogic() { return new RBCChamberORLogic(); }
RBCPatternLogic* createPatternLogic() { return new RBCPatternLogic(); }
TTUTrackingAlg* createTrackingAlg() { return new TTUTrackingAlg(); }
TTUSectorORLogic* createSectorORLogic() { return new TTUSectorORLogic(); }
TTUTwoORLogic* createTwoORLogic() { return new TTUTwoORLogic(); }
TTUWedgeORLogic* createWedgeORLogic() { return new TTUWedgeORLogic(); }
TTUPointingLogic* createPointingLogic() { return new TTUPointingLogic(); }
