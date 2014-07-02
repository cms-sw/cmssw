#ifndef CondFormats_L1TObjects_L1RPCConeDefinition_h
#define CondFormats_L1TObjects_L1RPCConeDefinition_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>

class L1RPCConeDefinition {



  public:
      //  For logplane sizes
    //typedef std::vector<int> TLogPlaneSize;
    //typedef std::vector<TLogPlaneSize > TLPSizesInTowers;
    struct TLPSize {
//      TLPSize(signed char t, signed char lp, unsigned char size) : 
//          m_tower(t), m_LP(lp), m_size(size) {};
      signed char m_tower;
      signed char m_LP;
      unsigned char m_size;
    
    COND_SERIALIZABLE;
};
    typedef std::vector<TLPSize> TLPSizeVec;

    
          
      // For (roll,hwplane)->tower mapping
      /*
    typedef std::vector<int> TTowerList;
    typedef std::vector<TTowerList > THWplaneToTower;
    typedef std::vector<THWplaneToTower > TRingsToTowers;*/

    struct TRingToTower {
//      TRingToTower(signed char ep, signed char hp, signed char t, unsigned char i) : 
//        m_etaPart(ep), m_hwPlane(hp), m_tower(t), m_index(i) {};
      signed char m_etaPart;
      signed char m_hwPlane;
      signed char m_tower;
      unsigned char m_index;
    
    COND_SERIALIZABLE;
};
    typedef std::vector<TRingToTower> TRingToTowerVec;
    
    // For (roll,hwplane)->logplane mapping
    
    /*
    typedef std::vector<int> TLPList;
    typedef std::vector<TLPList > THWplaneToLP;
    typedef std::vector<THWplaneToLP > TRingsToLP;
    */

    struct TRingToLP {
//      TRingToLP(signed char ep, signed char hp, signed char lp, unsigned char i) : 
//          m_etaPart(ep), m_hwPlane(hp), m_LP(lp), m_index(i) {};
      signed char m_etaPart;
      signed char m_hwPlane;
      signed char m_LP;
      unsigned char m_index;
    
    COND_SERIALIZABLE;
};
    typedef std::vector<TRingToLP> TRingToLPVec;

    
    

    //int getLPSize(int tower) const {return m_LPSizesInTowers.at(tower);};
    //const TLPSizesInTowers &  getLPSizes() const { return m_LPSizesInTowers;};

    void setFirstTower(int tow) {m_firstTower = tow;};
    void setLastTower(int tow) {m_lastTower = tow;};

    /*
    void setLPSizeForTowers(const TLPSizesInTowers & lpSizes) { m_LPSizesInTowers = lpSizes;};
    const TLPSizesInTowers & getLPSizeForTowers() const { return m_LPSizesInTowers;};
    */
    void setLPSizeVec(const TLPSizeVec & lpSizes) { m_LPSizeVec = lpSizes;};
    const TLPSizeVec & getLPSizeVec() const { return m_LPSizeVec;};
    
    /*
    void setRingsToTowers(const TRingsToTowers & RingsToTowers) { m_RingsToTowers = RingsToTowers;};
    const TRingsToTowers & getRingsToTowers() const { return m_RingsToTowers;};*/
  void setRingToTowerVec(const TRingToTowerVec & ringToTowerVec) 
        { m_ringToTowerVec = ringToTowerVec;};
  const TRingToTowerVec & getRingToTowerVec() const { return m_ringToTowerVec;};

    
  /*
    void setRingsToLP(const TRingsToLP & RingsToLP) {m_RingsToLP = RingsToLP;};
    const TRingsToLP & getRingsToLP() const {return m_RingsToLP;};
  */
  void setRingToLPVec(const TRingToLPVec & ringToLPVec) {m_ringToLPVec = ringToLPVec;};
  const TRingToLPVec & getRingToLPVec() const {return m_ringToLPVec;};

            
  private:
         
    int m_firstTower;
    int m_lastTower;
    //TLPSizesInTowers m_LPSizesInTowers;
    TLPSizeVec m_LPSizeVec;
    //TRingsToTowers m_RingsToTowers;
    TRingToTowerVec m_ringToTowerVec;
    
    
    //TRingsToLP m_RingsToLP;
    TRingToLPVec m_ringToLPVec;







  COND_SERIALIZABLE;
};



#endif
