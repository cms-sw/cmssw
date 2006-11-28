/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2004                                                      *
*                                                                              *
*******************************************************************************/
#ifndef L1RpcTriggerBoardH
#define L1RpcTriggerBoardH

/** \class RPCTriggerBoard
  * Class describing the Trigger Board. In hardware on one TB thera are m_PAC
  * procesors for 3 or 4 towers of one sector.
  * In this implementation the RPCTriggerBoard does not holds the m_PAC,
  * but acces them through RPCTriggerConfiguration, beacouse deifferent
  * configuration are possible (for example the same m_PAC for every Log Segment of given m_tower).
  * \author Karol Bunkowski (Warsaw)
  */

#include <vector>
#include <string>
#include "L1Trigger/RPCTrigger/src/RPCTBGhostBuster.h"
#include "L1Trigger/RPCTrigger/src/RPCTriggerConfiguration.h"
//---------------------------------------------------------------------------
class RPCTriggerBoard {
public:
  RPCTriggerBoard(RPCTBGhostBuster* tbGhostBuster,
                    RPCTriggerConfiguration* triggerConfig,
                    int tbNum);

  /** Runs RPCPac::run() for cone. Converts RPCPacMuon to RPCTBMuon
    * and puts it to the m_PacsMuonsVec. @return true if non-empty muon was return
    * by m_PAC*/
  bool runCone(const RPCLogCone& cone);

  /** Creates L1RpcTBMuonsVec2 from muons from m_PacsMuonsVec.
    * Then runs RPCTBGhostBuster::run().
    * @return 4 muons or empty vector. */
  L1RpcTBMuonsVec runTBGB();

private:

  int m_TBNumber; //!< 0...8 , 0 = tbn4, m_tower -16..-13, 4 = tb0

  RPCTriggerConfiguration* m_TriggerConfig;

  RPCTBGhostBuster* m_TBGhostBuster;

  L1RpcTBMuonsVec m_PacsMuonsVec;

};
#endif
