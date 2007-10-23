#ifndef L1Trigger_RPCDetInfo_h
#define L1Trigger_RPCDetInfo_h


/** \class RPCDetInfo
 *
 * \brief Class describing connections of RPC chamber to trigger system
 *  
 * \author Tomasz Fruboes
 * \todo check if defualt constructor must be somewhow special to be used with map
 *
 */



#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include <Geometry/RPCGeometry/interface/RPCGeometry.h> // To have RPCRoll. check if needed
#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include <Geometry/CommonTopologies/interface/RectangularStripTopology.h> // XXX
#include <Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h>

class RPCDetInfo{

  public:
    typedef std::map<int, float> RPCStripPhiMap;
    
    
    
    RPCDetInfo(){ }; // To be able to use map
    RPCDetInfo(RPCRoll* roll);

    uint32_t rawId();
    float getPhi();
    float getMinPhi();
    float getMaxPhi();
    float getEtaCentre();
    int getRingFromRollsId();
    //void setEtaMin(float);
    //void setEtaMax(float);
    int getMinTower();
    int getMaxTower();
    int getRegion();
    int getRing();
    int getHwPlane(); 
    int getRoll();
    int getGlobRollNo();

    RPCStripPhiMap getRPCStripPhiMap();
    //int giveNextStripInPhi(int);
    //int givePrevStripInPhi(int);
    int giveStripOfPhi(float);
        
    void printContents();
    int etaToTower(float eta);

  private:
    void setHwPlane(); 
    int etaToSign(float eta);
    //int getHwPlane(int region, int station, int layer);
    float transformPhi(float phi);
    void makeStripPhiMap(RPCRoll* roll);
  // Members
  
    
  private:

    uint32_t m_detId; ///< DetId number
    int m_region; ///< (+-)1 for endcaps, 0 for barrell
    int m_ring;   ///< Wheel number for barrell, Ring number for endcap (RE*/1 - RE*/3 -> 1...3 )
    int m_station;///< Muon station number (RB1...RB4; RE1...RE4)
    int m_layer;  ///< Inner or outer layer
    int m_roll;   ///< Roll number
    int m_sector; ///< Sector number
    int m_subsector; ///< Subsector number
    int m_hwPlane; ///< 1...6 for barell and 1...4 for endcaps
    float m_etaMin;  ///< etaMin and etaMax define to which m_tower(s) chamber contributes
    float m_etaMax;  ///< etaMin and etaMax define to which m_tower(s) chamber contributes
    float m_etaCentre; ///< eta of centre of this detId
    float m_phi;    ///< Phi of center of this detId (different than globalPoint.phi() - [0...2PI[)
    float m_phiMin; ///< The lowest phi of strips
    float m_phiMax; ///< The highest phi of strips
    int m_towerMin; ///< Lowest tower number to which chamber contributes
    int m_towerMax; ///< Highest tower number to which chamber contributes
    int m_globRoll; 
    static const float m_towerBounds[]; ///< Defines m_tower bounds
    
    
    RPCStripPhiMap m_stripPhiMap;
    
};
#endif
