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
// $Id: L1RPCConeBuilder.h,v 1.3 2008/04/16 13:44:11 fruboes Exp $
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
      
      

      // compressed connections
      struct TCompressedCon{
        signed char m_tower;
        signed char m_mul; 
        unsigned char m_PAC;
        unsigned char m_logplane;
        unsigned char m_validForStripFirst; 
        unsigned char m_validForStripLast; 
        signed short m_offset;
        
        TCompressedCon() : m_tower(99),  m_mul(99), m_PAC(0), 
          m_logplane(99), m_validForStripFirst(0), m_validForStripLast(0), m_offset(-1000){};
          
        int getLogStrip(int strip, const TLPSizesInTowers & LPSizesInTowers) const{
          int ret = -1;
          if ( strip >= m_validForStripFirst && strip <= m_validForStripLast ){ 
            ret = int(m_mul)*strip+int(m_offset);
          
            if (ret<0 || ret > LPSizesInTowers.at(std::abs(m_tower)).at(m_logplane-1)  )
              return -1;
          
          }
          return ret;
        }
        
        void addStrip(unsigned char strip) {
          if (m_validForStripFirst==0) {
            m_validForStripFirst = strip;
            m_validForStripLast = strip;
          } else if (strip < m_validForStripFirst){
            m_validForStripFirst = strip;
          }
          else if (strip > m_validForStripLast){
            m_validForStripLast = strip;
          }
        
        }
        
      };
      
      typedef std::vector<TCompressedCon> TCompressedConVec;
      typedef std::map<uint32_t, TCompressedConVec> TCompressedConMap;
      
      
      
      L1RPCConeBuilder();
      virtual ~L1RPCConeBuilder();

      void setLPSizeForTowers(const TLPSizesInTowers & lpSizes) { m_LPSizesInTowers = lpSizes;};
      void setRingsToTowers(const TRingsToTowers & RingsToTowers) { m_RingsToTowers = RingsToTowers;};
      void setRingsToLP(const TRingsToLP & RingsToLP) {m_RingsToLP = RingsToLP;};
            
      void setConeConnectionMap(const TConMap & connMap) { m_coneConnectionMap = connMap;};
      void setCompressedConeConnectionMap(const TCompressedConMap & cmpConnMap) 
          {  m_compressedConeConnectionMap = cmpConnMap;};
      
      std::pair<TStripConVec::const_iterator, TStripConVec::const_iterator> 
          getConVec(uint32_t det, unsigned char strip) const ;
      
      std::pair<TCompressedConVec::const_iterator, TCompressedConVec::const_iterator> 
          getCompConVec(uint32_t det) const ;
      
      void setFirstTower(int tow) {m_firstTower = tow;};
      void setLastTower(int tow) {m_lastTower = tow;};
      
      const TLPSizesInTowers &  getLPSizes() const { return m_LPSizesInTowers;};
      
   private:
         
      int m_firstTower;
      int m_lastTower;
      TLPSizesInTowers m_LPSizesInTowers;
      TRingsToTowers m_RingsToTowers;
      TRingsToLP m_RingsToLP;
      
      TConMap m_coneConnectionMap; 
      TCompressedConMap m_compressedConeConnectionMap;
};


#endif
