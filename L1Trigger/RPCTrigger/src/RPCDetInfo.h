/** \class RPCDetInfo
 *
 * Description: Class describing connections of RPC chamber to trigger system
 *  
 * Implementation:
  * \author Tomasz Fruboes
 *
 */
//TODO: Check if defualt constructor must be somewhow special (to use with map)

#ifndef RPCTrigger_RPCDetInfo_h
#define RPCTrigger_RPCDetInfo_h

#include "DataFormats/MuonDetId/interface/RPCDetId.h" // FIXME: included only to have uint32_t 


class RPCDetInfo{

  public:
    RPCDetInfo(){ }; // To be able to use map
    RPCDetInfo(uint32_t detId, int region, int ring, int station, int layer, int roll);
    
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
    // TODO do not use array
    static const float m_towerBounds[];
  
    
    uint32_t m_detId;
    int m_region; // +-1 for endcaps, 0 for barrell
    int m_ring;   // wheel number for barrell, Ring number for endcap (RE*/1 - RE*/3 -> 1...3 )
    int m_station;// muon station number (RB1...RB4 RE1...RE4) 
    int m_layer;  // inner or outer layer
    int m_roll;
    
    int m_hwPlane; // 1...6 for barell and 1...4 for endcaps
        

    float m_etaMin;  // etaMin and etaMax define to which tower(s) chamber 
    float m_etaMax;  // contributes. Current implementation is bad
        

    
};
#endif
