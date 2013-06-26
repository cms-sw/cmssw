#ifndef L1Trigger_RPCTBGhostBuster_h
#define L1Trigger_RPCTBGhostBuster_h

/** \class RPCTBGhostBuster
  * Peformes the Trigger Board Ghost Buster and sorter algorithms.
  * Because the class does not keep any data and GB is the same for every TB,
  * there might be one and the same object of this class for all TBs.
  * \author Karol Bunkowski (Warsaw)
  */

#include "L1Trigger/RPCTrigger/interface/RPCTBMuon.h"
//---------------------------------------------------------------------------
class RPCTBGhostBuster {
public:
   /** Calls gBPhi and gBEta.
    * @param pacMuonsVec2 pacMuonsVec2[0..3][0..11] (4 Towers x 12 Segments (PACs) ),
    * if pacMuonsVec2[i].size() == 0, means no nonempty muon.
    * @return always 4 muons.*/
  L1RpcTBMuonsVec run(L1RpcTBMuonsVec2 &pacMuonsVec2) const;

  /** @param pacMuonsVec pacMuonsVec.size() should be equal to 12 (Segment in given TB count)
    * or 0,
    * Performs Phi Goustbuster - between muons in one sector of the same m_tower.
    * @return if pacMuonsVec has size = 0, returns L1RpcTBMuonsVec with size 0,
    * otherwise returns vector of size = 4.
    * Calls RPCTBMuon::setPhiAddr() for each alive muon.*/
  L1RpcTBMuonsVec gBPhi(L1RpcTBMuonsVec &pacMuonsVec) const;

  /** @param gbPhiMuonsVec2 size() should be equal to 4 (max number of m_tower in TB).
    * Performs the Eta Goustbuster - between muons from diferent towers.
    * Calls RPCTBMuon::setPhiAddr() for each alive muon.
    * @return always 4 muons.*/
  L1RpcTBMuonsVec gBEta(L1RpcTBMuonsVec2 &gbPhiMuonsVec2) const;

};
#endif
