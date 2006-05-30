#ifndef RPCTrigger_RPCDetInfo_h
#define RPCTrigger_RPCDetInfo_h


/** \class RPCDetInfo
 *
 * \brief Class describing connections of RPC chamber to trigger system
 *  
 * \author Tomasz Fruboes
 * \todo Check if defualt constructor must be somewhow special to be used with map
 * \todo Store phi info of all strips
 *
 */



#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include <Geometry/RPCGeometry/interface/RPCGeometry.h> // To have RPCRoll. check if needed
#include <Geometry/Records/interface/MuonGeometryRecord.h>


#include <Geometry/CommonTopologies/interface/RectangularStripTopology.h> // XXX
#include <Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h>

class RPCDetInfo{

  public:
    RPCDetInfo(){ }; // To be able to use map
    RPCDetInfo(RPCRoll* roll);

    uint32_t rawId();
    int getCurlId();
    void setEtaMin(float);
    void setEtaMax(float);
    
  private:
    int etaToTower(float eta);
    int etaToSign(float eta);
    //int getHwPlane(int region, int station, int layer);
    int getHwPlane(); 
    
  // Members
  
    
  private:

    uint32_t m_detId; ///< DetId number
    int m_region; ///< (+-)1 for endcaps, 0 for barrell
    int m_ring;   ///< Wheel number for barrell, Ring number for endcap (RE*/1 - RE*/3 -> 1...3 )
    int m_station;///< Muon station number (RB1...RB4; RE1...RE4)
    int m_layer;  ///< Inner or outer layer
    int m_roll;   ///< Roll number
    int m_hwPlane; ///< 1...6 for barell and 1...4 for endcaps
    float m_etaMin;  ///< etaMin and etaMax define to which tower(s) chamber contributes
    float m_etaMax;  ///< etaMin and etaMax define to which tower(s) chamber contributes
    static const float m_towerBounds[]; ///< Defines tower bounds
    
};
#endif
