#ifndef CondFormats_RPCObjects_L1RPCConeBuilder_h
#define CondFormats_RPCObjects_L1RPCConeBuilder_h
// -*- C++ -*-
//
// Package:     RPCObjects
// Class  :     L1RPCConeBuilder
// 
/**\class L1RPCConeBuilder L1RPCConeBuilder.h CondFormats/RPCObjects/interface/L1RPCConeBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Tomasz Fruboes
//         Created:  Fri Feb 22 12:27:02 CET 2008
// $Id: L1RPCConeBuilder.h,v 1.2 2008/03/14 12:38:20 fruboes Exp $
//

#include <vector>
#include <map>

//#include "CondFormats/RPCObjects/interface/RPCConeConnection.h"



class L1RPCConeBuilder
{
   
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

      // uncompressed connections
      struct TStripCon{
        signed char m_tower;
        unsigned char m_PAC;
        unsigned char m_logplane;
        unsigned char m_logstrip;
      };
      
      typedef std::vector<TStripCon> TStripConVec;
      typedef std::map<unsigned char, TStripConVec> TStrip2ConVec;
      typedef std::map<uint32_t, TStrip2ConVec> TConMap;
      
      L1RPCConeBuilder();
      virtual ~L1RPCConeBuilder();

      void setLPSizeForTowers(const TLPSizesInTowers & lpSizes) { m_LPSizesInTowers = lpSizes;};
      void setRingsToTowers(const TRingsToTowers & RingsToTowers) { m_RingsToTowers = RingsToTowers;};
      void setRingsToLP(const TRingsToLP & RingsToLP) {m_RingsToLP = RingsToLP;};
            
      void setConeConnectionMap(const TConMap & connMap) { m_coneConnectionMap = connMap;};
      //const TConMap & getConeConnectionMap() const { return m_coneConnectionMap;};
      
      /*
      const TStripConVec & getConVec(uint32_t det, unsigned char strip) const 
          { 
            //return const_cast<TConMap>(m_coneConnectionMap)[det][strip];
            return m_coneConnectionMap[det][strip];
      };*/
      
      std::pair<TStripConVec::const_iterator, TStripConVec::const_iterator> 
          getConVec(uint32_t det, unsigned char strip) const ;
//      {
//       return std::make_pair(m_coneConnectionMap[det][strip].begin(),m_coneConnectionMap[det][strip].end());
//      };
      void setFirstTower(int tow) {m_firstTower = tow;};
      void setLastTower(int tow) {m_lastTower = tow;};
      
      const TLPSizesInTowers &  getLPSizes() const { return m_LPSizesInTowers;};
      
   private:
//      L1RPCConeBuilder(const L1RPCConeBuilder&); // stop default

//      const L1RPCConeBuilder& operator=(const L1RPCConeBuilder&); // stop default

      // ---------- member data --------------------------------
      
//      const short m_towerCount;
      //const short m_logplaneCount;
         
      int m_firstTower;
      int m_lastTower;
      TLPSizesInTowers m_LPSizesInTowers;
      TRingsToTowers m_RingsToTowers;
      TRingsToLP m_RingsToLP;
      
      TConMap m_coneConnectionMap; // mutable needed for std::map [] operator :( ; should be better way
};


#endif
