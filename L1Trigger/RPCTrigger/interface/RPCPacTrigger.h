#ifndef L1Trigger_RPCPacTrigger_h
#define L1Trigger_RPCPacTrigger_h

/** \class RPCPacTrigger
  * Simulations of Pattern Comparator trigger. For every event runs PACs, ghost-busters and sorters.
  * Contains the structure of hardware: Trigger Crates, Final Sorters, etc.
  * \author Karol Bunkowski (Warsaw) */

#include "L1Trigger/RPCTrigger/interface/RPCLogCone.h"
#include "L1Trigger/RPCTrigger/interface/RPCTriggerCrate.h"
#include "L1Trigger/RPCTrigger/interface/RPCFinalSorter.h"
#include "L1Trigger/RPCTrigger/interface/RPCHalfSorter.h"
#include "L1Trigger/RPCTrigger/interface/RPCTriggerConfiguration.h"

#include <FWCore/Framework/interface/ESHandle.h>
#include "CondFormats/L1TObjects/interface/L1RPCHsbConfig.h"

class RPCPacTrigger {
public:
  /** Construct the structure of Trigger Crates,
    * m_TBGhostBuster, m_TCGhostBusterSorter and m_FinalSorter are also created.
    * Since these object do not holds any data, on instance is enough for whole system.
    * Found muons are stored in TB and TC. */
  RPCPacTrigger(RPCTriggerConfiguration* triggerConfig);

  /** For every logCone from logConesVec runs RPCTriggerCrate::runCone().
    * Then for every TC runs RPCTriggerCrate::runTCGBSorter().
    * Then runs and RPCFinalSorter::run().
    * @return m_GBFinalMuons[be][iMU] be = 0 = barrel; be = 1 = endcaps,
    * 4 muons from barrel and 4 muons from endcaps, (some can be empty)*/
  L1RpcTBMuonsVec2 runEvent(const L1RpcLogConesVec& logConesVec, edm::ESHandle<L1RPCHsbConfig> hsbConf );


  /** Returns vector of not empty muons.
    * Can be called only after runEvent() */
  L1RpcTBMuonsVec getNotEmptyMuons();

private:
//-------------hardware------------------
  std::vector<RPCTriggerCrate> m_TriggerCratesVec;

  RPCTriggerConfiguration* m_TrigCnfg;

  RPCFinalSorter m_FinalSorter;
  RPCHalfSorter m_HalfSorters;
  
  //---------------------------------------

  /** Muons found in each event. m_GBFinalMuons[be][iMU] be = 0 = barrel; be = 1 = endcaps,
    * 4 muons from barrel and 4 muons from endcaps, (some can be empty)*/
  L1RpcTBMuonsVec2 m_GBFinalMuons;
};
#endif
