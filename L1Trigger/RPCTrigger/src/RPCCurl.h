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
    void printContents() const;

  private:
    // Not filled by now
    int m_TowerMin; ///< The lowest tower no. to which curl contributes
    int m_TowerMax; ///< The highest tower no. to which curl contributes
    int m_HardwarePlane; ///< Hardware plane no.
    int m_Region; ///< Region no - 0 for barell +-1 for endcaps
    int m_Wheel;  ///< Wheel number for barell, ring number for endcaps
    
    bool m_IsReferencePlane;  ///< tells if detIds from this curl form a ference plane
    // void giveTowerMin(float eta);
    // void giveTowerMax(float eta);

    typedef std::map<uint32_t, RPCDetInfo> RPCDetInfoMap;
    RPCDetInfoMap mRPCDetInfoMap; ///< Stores all DetId`s of a curl
    
};
#endif
