#ifndef RPCTrigger_RPCCurl_h
#define RPCTrigger_RPCCurl_h

/** \class RPCCurl
 *
 * \brief Class describng 2PI "rings" constructed from RpcDetId's of the same eta (same as L1RpcRoll in ORCA)
 * \author Tomasz Fruboes
 * \todo Sort RPCDetId`s in phi order
 *
 */

#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include "L1Trigger/RPCTrigger/src/RPCDetInfo.h"


class RPCCurl {
  public:
    RPCCurl();
    ~RPCCurl();
    bool isReferencePlane();
    bool addDetId(RPCDetInfo detInfo);
    void printContents() ;

  private:
    int m_towerMin; ///< The lowest tower no. to which curl contributes
    int m_towerMax; ///< The highest tower no. to which curl contributes
    int m_hardwarePlane; ///< Hardware plane no.
    int m_region; ///< Region no - 0 for barell +-1 for endcaps
    int m_ring;  ///< Wheel number for barell, ring number for endcaps
    int m_roll;  ///< roll no
    
    
    bool m_isDataFresh; ///< Defines if m_towerMin (Max) have real world contents
    bool m_isReferencePlane;  ///< tells if detIds from this curl form a ference plane
    // void giveTowerMin(float eta);
    // void giveTowerMax(float eta);
    
    
    typedef std::map<uint32_t, RPCDetInfo> RPCDetInfoMap;
    RPCDetInfoMap m_RPCDetInfoMap; ///< Stores all DetId`s of a curl
    
    //typedef std::map<float, uint32_t> RPCDetInfoPhiMap;
    typedef std::map<float, uint32_t> RPCDetInfoPhiMap;
    RPCDetInfoPhiMap m_RPCDetPhiMap; ///< Stores DetId`s rawId in a phi order.
    
    
};
#endif
