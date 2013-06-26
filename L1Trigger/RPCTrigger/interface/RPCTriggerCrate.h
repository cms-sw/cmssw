#ifndef L1Trigger_RPCTriggerCrate_h
#define L1Trigger_RPCTriggerCrate_h
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

#include "L1Trigger/RPCTrigger/interface/RPCTBMuon.h"
#include "L1Trigger/RPCTrigger/interface/RPCTriggerBoard.h"
#include "L1Trigger/RPCTrigger/interface/RPCTCGhostBusterSorter.h"

//---------------------------------------------------------------------------
class RPCTriggerCrate {
public:
  RPCTriggerCrate(RPCTriggerConfiguration* triggerConfig, int tcNum);

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

  RPCTCGhostBusterSorter m_TCGhostBusterSorter;

  bool m_WasMuon;
};
#endif
