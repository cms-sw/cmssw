/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2004                                                      *
*                                                                              *
*******************************************************************************/
#ifndef L1RpcPacTriggerH
#define L1RpcPacTriggerH

/** \class L1RpcPacTrigger
  * Simulations of Pattern Comparator trigger. For every event runs PACs, ghost-busters and sorters.
  * Contains the structure of hardware: Trigger Crates, Final Sorters, etc.
  * \author Karol Bunkowski (Warsaw) */

#include "L1Trigger/RPCTrigger/src/L1RpcLogCone.h"
#include "L1Trigger/RPCTrigger/src/L1RpcTriggerCrate.h"
#include "L1Trigger/RPCTrigger/src/L1RpcFinalSorter.h"
#include "L1Trigger/RPCTrigger/src/L1RpcTriggerConfiguration.h"

class L1RpcPacTrigger {
public:
  /** Construct the structure of Trigger Crates,
    * TBGhostBuster, TCGhostBusterSorter and FinalSorter are also created.
    * Since these object do not holds any data, on instance is enough for whole system.
    * Found muons are stored in TB and TC. */
  L1RpcPacTrigger(L1RpcTriggerConfiguration* triggerConfig);

  ~L1RpcPacTrigger() { }

  /** For every logCone from logConesVec runs L1RpcTriggerCrate::RunCone().
    * Then for every TC runs L1RpcTriggerCrate::RunTCGBSorter().
    * Then runs and L1RpcFinalSorter::Run().
    * @return GBFinalMuons[be][iMU] be = 0 = barrel; be = 1 = endcaps,
    * 4 muons from barrel and 4 muons from endcaps, (some can be empty)*/
  L1RpcTBMuonsVec2 RunEvent(const L1RpcLogConesVec& logConesVec);


  /** Returns vector of not empty muons.
    * Can be called only after RunEvent() */
  L1RpcTBMuonsVec GetNotEmptyMuons();

private:
//-------------hardware------------------
  std::vector<L1RpcTriggerCrate> TriggerCratesVec;

  L1RpcTriggerConfiguration* TrigCnfg;

  L1RpcFinalSorter FinalSorter;

  /** One TCGhostBusterSorter, the same for every TC.
    * (In hardware there is 12 TCGhostBusterSorters, one per each TC.
    * TriggerCrates are constructed with pointer to this object. */
  L1RpcTCGhostBusterSorter TCGhostBusterSorter;

  /** One TBGhostBuster, the same for every TB.
    * (In hardware thery is one per each TB.
    * TB are constructed with pointer to this object. */
  L1RpcTBGhostBuster TBGhostBuster;

//---------------------------------------

  /** Muons found in each event. GBFinalMuons[be][iMU] be = 0 = barrel; be = 1 = endcaps,
    * 4 muons from barrel and 4 muons from endcaps, (some can be empty)*/
  L1RpcTBMuonsVec2 GBFinalMuons;
};
#endif
