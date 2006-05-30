#ifndef RPCTrigger_RPCTriggerGeo_h
#define RPCTrigger_RPCTriggerGeo_h

/** \class RPCTriggerGeo
 *
 * \brief Class describing RPC trigger geometry
 *  
 * Aim: easly convert RPCdetId.firedStrip to loghit/logcone
 * \author Tomasz Fruboes 
 *
 */

#include <FWCore/Framework/interface/ESHandle.h> // Handle to read geometry

#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include "L1Trigger/RPCTrigger/src/RPCDetInfo.h"
#include "L1Trigger/RPCTrigger/src/RPCCurl.h"

class RPCTriggerGeo {
  public:
    RPCTriggerGeo();
    ~RPCTriggerGeo() {};
    
    void buildGeometry(edm::ESHandle<RPCGeometry> rpcGeom);
    bool isGeometryBuilt();
  


  private:
    void addDet(RPCRoll* roll);
    void printCurlMapInfo();
    int etaToTower(float eta);
                              
    bool m_isGeometryBuilt; ///< Determines if geometry is built allready
    typedef std::map<uint32_t, RPCCurl> RPCCurlMap; 
    RPCCurlMap m_RPCCurlMap; ///< Stores all curls
    
};
#endif
