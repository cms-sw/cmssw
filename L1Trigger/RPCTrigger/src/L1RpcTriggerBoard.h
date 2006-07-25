/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2004                                                      *
*                                                                              *
*******************************************************************************/
#ifndef L1RpcTriggerBoardH
#define L1RpcTriggerBoardH

/** \class L1RpcTriggerBoard
  * Class describing the Trigger Board. In hardware on one TB thera are PAC
  * procesors for 3 or 4 towers of one sector.
  * In this implementation the L1RpcTriggerBoard does not holds the PAC,
  * but acces them through L1RpcTriggerConfiguration, beacouse deifferent
  * configuration are possible (for example the same PAC for every Log Segment of given tower).
  * \author Karol Bunkowski (Warsaw)
  */

#include <vector>
#include <string>
#include "L1Trigger/RPCTrigger/src/L1RpcTBGhostBuster.h"
#include "L1Trigger/RPCTrigger/src/L1RpcTriggerConfiguration.h"
//---------------------------------------------------------------------------
class L1RpcTriggerBoard {
public:
  L1RpcTriggerBoard(L1RpcTBGhostBuster* tbGhostBuster,
                    L1RpcTriggerConfiguration* triggerConfig,
                    int tbNum);

  /** Runs L1RpcPac::Run() for cone. Converts L1RpcPacMuon to L1RpcTBMuon
    * and puts it to the PacsMuonsVec. @return true if non-empty muon was return
    * by PAC*/
  bool RunCone(const L1RpcLogCone& cone);

  /** Creates L1RpcTBMuonsVec2 from muons from PacsMuonsVec.
    * Then runs L1RpcTBGhostBuster::Run().
    * @return 4 muons or empty vector. */
  L1RpcTBMuonsVec RunTBGB();

private:

  int TBNumber; //!< 0...8 , 0 = tbn4, tower -16..-13, 4 = tb0

  L1RpcTriggerConfiguration* TriggerConfig;

  L1RpcTBGhostBuster* TBGhostBuster;

  L1RpcTBMuonsVec PacsMuonsVec;

};
#endif
