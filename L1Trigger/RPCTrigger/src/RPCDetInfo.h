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
    int getCurlId();
    RPCDetInfo(){ }; 
    RPCDetInfo(uint32_t mDetId, int region, int ring, int station, int layer, int roll);
    uint32_t rawId();
        
  private:
    int etaToTower(float eta);
    int etaToSign(float eta);
    //int getHwPlane(int region, int station, int layer);
    int getHwPlane(); 
    
    // Members
  public:
    float mEtaMin;  // etaMin and etaMax define to which tower(s) chamber 
    float mEtaMax;  // contributes. Current implementation is bad
    
  private:
    static const float mTowerBounds[];
  
    uint32_t mDetId;
    int mRegion; // +-1 for endcaps, 0 for barrell
    int mRing;   // wheel number for barrell, Ring number for endcap (RE*/1 - RE*/3 -> 1...3 )
    int mStation;// muon station number (RB1...RB4 RE1...RE4) 
    int mLayer;  // inner or outer layer
    int mRoll;
    
    int mHwPlane; // 1...6 for barell and 1...4 for endcaps
        

    
    
    //int mRing;   // Wheel number in barell, disk number in endcap (=rpcdetid::ring())
                   // Note: endcap numbering is non-intutive (123-----123)
    //int mRegion; // +-1 for endcap, 0 for barell

    
};
#endif
