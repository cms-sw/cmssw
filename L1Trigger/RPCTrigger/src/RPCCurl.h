/** \class RPCCurl
 *
 * Description: Class describng 2PI "rings" constructed from RpcDetId's of the same eta (same as L1RpcRoll in ORCA)
 *  
 * Implementation:
 *  
 * \author Tomasz Fruboes 
 *
 */

#ifndef RPCTrigger_RPCCurl_h
#define RPCTrigger_RPCCurl_h


#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include "L1Trigger/RPCTrigger/src/RPCDetInfo.h"


class RPCCurl {
  public:
    RPCCurl();
    ~RPCCurl();
    bool isReferencePlane();

    bool addDetId(RPCDetInfo detInfo);  // Returns true if detId was succesfully added to the curl
  
    void printContents() const;

  private:
    // Not filled by now
    int m_TowerMin;
    int m_TowerMax;
    int m_HardwarePlane;
    int m_Region;
    int m_Wheel;
    
    bool m_IsReferencePlane;
    // void giveTowerMin(float eta);
    // void giveTowerMax(float eta);

    typedef std::map<uint32_t, RPCDetInfo> RPCDetInfoMap;
    RPCDetInfoMap mRPCDetInfoMap;
    
};
#endif
