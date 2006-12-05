#ifndef L1RpcTriggerCrateH
#define L1RpcTriggerCrateH
//#define LOCALDEBUG
/** \class RPCTriggerCrate
  * Class describing the Trigger Crate. In one Trigger Crate thera are
  * Trigger Boards fro one sector.
  * \author Karol Bunkowski (Warsaw)
  */

#ifndef _STAND_ALONE
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#endif // _STAND_ALONE

#include <vector>

#include "L1Trigger/RPCTrigger/src/RPCTBMuon.h"
#include "L1Trigger/RPCTrigger/src/RPCTriggerBoard.h"
#include "L1Trigger/RPCTrigger/src/RPCTCGhostBusterSorter.h"

//---------------------------------------------------------------------------
class RPCTriggerCrate {
public:
  RPCTriggerCrate(RPCTCGhostBusterSorter* tcGhostBusterSorter, 
                    RPCTBGhostBuster* tbGhostBuster,
                    RPCTriggerConfiguration* triggerConfig,
                    int tcNum);

  /** Runs RPCTriggerBoard::runCone() for every TB. Cheks, if any non empty muons were found*/
  void runCone(const RPCLogCone& cone);

  /** If in runCone() any muons were found,
    * runs RPCTriggerBoard::runTBGB() for every TB.
    * Then runs RPCTCGhostBusterSorter::run(). */
  L1RpcTBMuonsVec runTCGBSorter();

private:
  int m_TCNumber; //!< The number of this TriggerCrate (0 - 11)

  RPCTriggerConfiguration* m_TriggerConfig;

  std::vector<RPCTriggerBoard> m_TBsVec; //!< Here TB belonging to thie TC are stored.

  RPCTCGhostBusterSorter* m_TCGhostBusterSorter; //!< Should point to the object kept by RPCPacTrigger.

  bool m_WasMuon;
};
#endif
