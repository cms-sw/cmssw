/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2004                                                      *
*                                                                              *
*******************************************************************************/

#ifndef L1RpcTriggerCrateH
#define L1RpcTriggerCrateH
//#define LOCALDEBUG
/** \class L1RpcTriggerCrate
  * Class describing the Trigger Crate. In one Trigger Crate thera are
  * Trigger Boards fro one sector.
  * \author Karol Bunkowski (Warsaw)
  */

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>

#include "L1Trigger/RPCTrigger/src/L1RpcTBMuon.h"
#include "L1Trigger/RPCTrigger/src/L1RpcTriggerBoard.h"
#include "L1Trigger/RPCTrigger/src/L1RpcTCGhostBusterSorter.h"

//---------------------------------------------------------------------------
class L1RpcTriggerCrate {
public:
  L1RpcTriggerCrate(L1RpcTCGhostBusterSorter* tcGhostBusterSorter, L1RpcTBGhostBuster* tbGhostBuster, L1RpcTriggerConfiguration* triggerConfig, int tcNum);

  /** Runs L1RpcTriggerBoard::RunCone() for every TB. Cheks, if any non empty muons were found*/
  void RunCone(const L1RpcLogCone& cone) {
    
    /*
    RPCParam::L1RpcConeCrdnts ConeCrdnts = cone.GetConeCrdnts();
    LogDebug("RPCTrigger") << " Tower: " << ConeCrdnts.Tower
        << " LSector" << ConeCrdnts.LogSector
        << " LSegment" << ConeCrdnts.LogSegment
        << " TBNum:" << TriggerConfig->GetTBNum(cone.GetConeCrdnts())<< std::endl;
    */
    if(TBsVec[TriggerConfig->GetTBNum(cone.GetConeCrdnts())].RunCone(cone) )
      WasMuon = true;
  };

  /** If in RunCone() any muons were found,
    * runs L1RpcTriggerBoard::RunTBGB() for every TB.
    * Then runs L1RpcTCGhostBusterSorter::Run(). */
  L1RpcTBMuonsVec RunTCGBSorter();

private:
  int TCNumber; //!< The number of this TriggerCrate (0 - 11)

  L1RpcTriggerConfiguration* TriggerConfig;

  std::vector<L1RpcTriggerBoard> TBsVec; //!< Here TB belonging to thie TC are stored.

  L1RpcTCGhostBusterSorter* TCGhostBusterSorter; //!< Should point to the object kept by L1RpcPacTrigger.

  bool WasMuon;
};
#endif
