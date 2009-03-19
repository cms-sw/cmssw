#ifndef CondFormats_RPCObjects_L1RPCConeDefinition_h
#define CondFormats_RPCObjects_L1RPCConeDefinition_h

#include <vector>

class L1RPCConeDefinition {



  public:
      //  For logplane sizes
    typedef std::vector<int> TLogPlaneSize;
    typedef std::vector<TLogPlaneSize > TLPSizesInTowers;
      
      // For (roll,hwplane)->tower mapping
    typedef std::vector<int> TTowerList;
    typedef std::vector<TTowerList > THWplaneToTower;
    typedef std::vector<THWplaneToTower > TRingsToTowers;

      
      // For (roll,hwplane)->logplane mapping
    typedef std::vector<int> TLPList;
    typedef std::vector<TTowerList > THWplaneToLP;
    typedef std::vector<THWplaneToTower > TRingsToLP;


    const TLPSizesInTowers &  getLPSizes() const { return m_LPSizesInTowers;};

    void setFirstTower(int tow) {m_firstTower = tow;};
    void setLastTower(int tow) {m_lastTower = tow;};

    void setLPSizeForTowers(const TLPSizesInTowers & lpSizes) { m_LPSizesInTowers = lpSizes;};
    const TLPSizesInTowers & getLPSizeForTowers() const { return m_LPSizesInTowers;};
    
    void setRingsToTowers(const TRingsToTowers & RingsToTowers) { m_RingsToTowers = RingsToTowers;};
    const TRingsToTowers & getRingsToTowers() const { return m_RingsToTowers;};
    
    void setRingsToLP(const TRingsToLP & RingsToLP) {m_RingsToLP = RingsToLP;};
    const TRingsToLP & getRingsToLP() const {return m_RingsToLP;};

            
  private:
         
    int m_firstTower;
    int m_lastTower;
    TLPSizesInTowers m_LPSizesInTowers;
    TRingsToTowers m_RingsToTowers;
    TRingsToLP m_RingsToLP;






};



#endif
