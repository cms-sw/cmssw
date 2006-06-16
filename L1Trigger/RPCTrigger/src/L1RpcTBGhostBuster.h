/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2004                                                      *
*                                                                              *
*******************************************************************************/
#ifndef L1RpcTBGhostBusterH
#define L1RpcTBGhostBusterH

/** \class L1RpcTBGhostBuster
  * Peformes the Trigger Board Ghost Buster and sorter algorithms.
  * Because the class does not keep any data and GB is the same for every TB,
  * there might be one and the same object of this class for all TBs.
  * \author Karol Bunkowski (Warsaw)
  */

#include "L1Trigger/RPCTrigger/src/L1RpcTBMuon.h"
//---------------------------------------------------------------------------
class L1RpcTBGhostBuster {
public:
  /** Calls GBPhi and GBEta.
    * @param pacMuonsVec2 pacMuonsVec2[0..3][0..11] (4 Towers x 12 Segments (PACs) ),
    * if pacMuonsVec2[i].size() == 0, means no nonempty muon.
    * @return always 4 muons.*/
  L1RpcTBMuonsVec Run(L1RpcTBMuonsVec2 &pacMuonsVec2) const;

  /** @param pacMuonsVec pacMuonsVec.size() should be equal to 12 (Segment in given TB count)
    * or 0,
    * Performs Phi Goustbuster - between muons in one sector of the same tower.
    * @return if pacMuonsVec has size = 0, returns L1RpcTBMuonsVec with size 0,
    * otherwise returns vector of size = 4.
    * Calls L1RpcTBMuon::SetPhiAddr() for each alive muon.*/
  L1RpcTBMuonsVec GBPhi(L1RpcTBMuonsVec &pacMuonsVec) const;

  /** @param gbPhiMuonsVec2 size() should be equal to 4 (max number of tower in TB).
    * Performs the Eta Goustbuster - between muons from diferent towers.
    * Calls L1RpcTBMuon::SetPhiAddr() for each alive muon.
    * @return always 4 muons.*/
  L1RpcTBMuonsVec GBEta(L1RpcTBMuonsVec2 &gbPhiMuonsVec2) const;

private:

};
#endif
