#ifndef CondFormats_RPCObjects_L1RPCConeDefinition_h
#define CondFormats_RPCObjects_L1RPCConeDefinition_h

#include <vector>

class L1RPCConeDefinition {



  public:
      //  For logplane sizes
    typedef std::vector<int> TLogPlaneSize;
    typedef std::vector<TLogPlaneSize > TLPSizesInTowers;
      
      // For (roll,hwplane)->tower mapping
      /*
    typedef std::vector<int> TTowerList;
    typedef std::vector<TTowerList > THWplaneToTower;
    typedef std::vector<THWplaneToTower > TRingsToTowers;*/

    struct TRingToTower {
      TRingToTower(signed char ep, signed char hp, signed char t, unsigned char i) : 
        m_etaPart(ep), m_hwPlane(hp), m_tower(t), m_index(i) {};
      signed char m_etaPart;
      signed char m_hwPlane;
      signed char m_tower;
      unsigned char m_index;
    };
    typedef std::vector<TRingToTower> TRingToTowerVec;
    
    // For (roll,hwplane)->logplane mapping
    
    
    typedef std::vector<int> TLPList;
    typedef std::vector<TLPList > THWplaneToLP;
    typedef std::vector<THWplaneToLP > TRingsToLP;
    
    /*
    struct TRingToTower {
      TRingToTower(signed char ep, signed char hp, signed char t, unsigned char i) : 
          m_etaPart(ep), m_hwPlane(hp), m_tower(t), m_index(i) {};
      signed char m_etaPart;
      signed char m_hwPlane;
      signed char m_tower;
      unsigned char m_index;
    };
    typedef std::vector<TRingToTower> TRingToTowerVec;
    */
    
    

    //int getLPSize(int tower) const {return m_LPSizesInTowers.at(tower);};
    const TLPSizesInTowers &  getLPSizes() const { return m_LPSizesInTowers;};

    void setFirstTower(int tow) {m_firstTower = tow;};
    void setLastTower(int tow) {m_lastTower = tow;};

    void setLPSizeForTowers(const TLPSizesInTowers & lpSizes) { m_LPSizesInTowers = lpSizes;};
    const TLPSizesInTowers & getLPSizeForTowers() const { return m_LPSizesInTowers;};
    
    
    /*
    void setRingsToTowers(const TRingsToTowers & RingsToTowers) { m_RingsToTowers = RingsToTowers;};
    const TRingsToTowers & getRingsToTowers() const { return m_RingsToTowers;};*/
  void setRingToTowerVec(const TRingToTowerVec & ringToTowerVec) 
        { m_ringToTowerVec = ringToTowerVec;};
  const TRingToTowerVec & getRingToTowerVec() const { return m_ringToTowerVec;};

    
    void setRingsToLP(const TRingsToLP & RingsToLP) {m_RingsToLP = RingsToLP;};
    const TRingsToLP & getRingsToLP() const {return m_RingsToLP;};

            
  private:
         
    int m_firstTower;
    int m_lastTower;
    TLPSizesInTowers m_LPSizesInTowers;
    //TRingsToTowers m_RingsToTowers;
    TRingToTowerVec m_ringToTowerVec;
    TRingsToLP m_RingsToLP;






};



#endif
