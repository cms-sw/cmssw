//---------------------------------------------------------------------------

#ifndef L1RpcFinalSorterH
#define L1RpcFinalSorterH

/** \class L1RpcFinalSorter
  * Peformes the FinalSorter Ghost Buster and sorter algorithms.
  * Because the class does not keep any data and GB is the same for every TB,
  * there might be one and the same object of this class for all TBs.
  * \author Karol Bunkowski (Warsaw)
  */

#include "L1Trigger/RPCTrigger/src/L1RpcTBMuon.h"
#include "L1Trigger/RPCTrigger/src/L1RpcTriggerConfiguration.h"
//---------------------------------------------------------------------------

class L1RpcFinalSorter {
public:
  //L1RpcFinalSorter() {};

  L1RpcFinalSorter(L1RpcTriggerConfiguration* triggerConfig) {
    TrigCnfg = triggerConfig;

    GBOutputMuons.assign(2, L1RpcTBMuonsVec() );
  }

  /** Runs GB algorithm for half of the detector - 6 TC (sectors).
    * @return 4 munons from barrel (GBOutputMuons[0]),
    * and 4 from endcaps (GBOutputMuons[1]).*/
  L1RpcTBMuonsVec2 RunHalf(L1RpcTBMuonsVec2 &tcsMuonsVec2);

  //GBOutputMuons[be][iMU] , be = 0 = barrel; be = 1 = endcap
  //on input tower number (eta addr) continous (0 - 32, tower 0 = 16)
  //on output tower number (eta addr) 2'complement

  void RunFinalSorter(L1RpcTBMuonsVec2 &finalMuons);

  /** Runs RunHalf() for 2 detecors parts.
    * Converts tower number (eta addr) from continous (0 - 32, tower 0 = 16)
    * to 2'complement.
    * @return 4 munons from barrel (GBOutputMuons[0]),
    * and 4 from endcaps (GBOutputMuons[1]).*/
  L1RpcTBMuonsVec2 Run(L1RpcTBMuonsVec2 &tcsMuonsVec2);

private:
  L1RpcTBMuonsVec2 GBOutputMuons;

  L1RpcTriggerConfiguration* TrigCnfg;
  //GBOutputMuons[be][iMU] , be = 0 = barrel; be = 1 = endcap
};
#endif
