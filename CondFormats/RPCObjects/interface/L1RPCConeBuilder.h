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
// $Id: L1RPCConeBuilder.h,v 1.8 2009/03/20 15:10:53 michals Exp $
//

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <map>
#include <cstdint>
#include <cstdlib>
#include "CondFormats/L1TObjects/interface/L1RPCConeDefinition.h"
#include <memory>
#include <boost/serialization/shared_ptr.hpp>

class L1RPCConeBuilder {
public:
  // uncompressed connections
  struct TStripCon {
    signed char m_tower;
    unsigned char m_PAC;
    unsigned char m_logplane;
    unsigned char m_logstrip;

    COND_SERIALIZABLE;
  };
  typedef std::vector<TStripCon> TStripConVec;
  typedef std::map<unsigned char, TStripConVec> TStrip2ConVec;
  typedef std::map<uint32_t, TStrip2ConVec> TConMap;

  // compressed connections
  struct TCompressedCon {
    signed char m_tower;
    signed char m_mul;
    unsigned char m_PAC;
    unsigned char m_logplane;
    unsigned char m_validForStripFirst;
    unsigned char m_validForStripLast;
    signed short m_offset;

    TCompressedCon()
        : m_tower(99),
          m_mul(99),
          m_PAC(0),
          m_logplane(99),
          m_validForStripFirst(0),
          m_validForStripLast(0),
          m_offset(-1000){};

    int getLogStrip(int strip, const L1RPCConeDefinition::TLPSizeVec& LPSizeVec) const {
      int ret = -1;
      if (strip >= m_validForStripFirst && strip <= m_validForStripLast) {
        ret = int(m_mul) * strip + int(m_offset);

        int lpSize = -1;
        L1RPCConeDefinition::TLPSizeVec::const_iterator it = LPSizeVec.begin();
        L1RPCConeDefinition::TLPSizeVec::const_iterator itEnd = LPSizeVec.end();
        for (; it != itEnd; ++it) {
          if (it->m_tower != std::abs(m_tower) || it->m_LP != m_logplane - 1)
            continue;
          lpSize = it->m_size;
        }

        //FIXME
        if (lpSize == -1) {
          //throw cms::Exception("getLogStrip") << " lpSize==-1\n";
        }

        //if (ret<0 || ret > LPSizesInTowers.at(std::abs(m_tower)).at(m_logplane-1)  )
        if (ret < 0 || ret > lpSize)
          return -1;
      }
      return ret;
    }

    void addStrip(unsigned char strip) {
      if (m_validForStripFirst == 0) {
        m_validForStripFirst = strip;
        m_validForStripLast = strip;
      } else if (strip < m_validForStripFirst) {
        m_validForStripFirst = strip;
      } else if (strip > m_validForStripLast) {
        m_validForStripLast = strip;
      }
    }

    COND_SERIALIZABLE;
  };

  typedef std::vector<TCompressedCon> TCompressedConVec;
  typedef std::map<uint32_t, TCompressedConVec> TCompressedConMap;

  L1RPCConeBuilder();
  virtual ~L1RPCConeBuilder();

  void setConeConnectionMap(const std::shared_ptr<TConMap> connMap) { m_coneConnectionMap = connMap; };

  void setCompressedConeConnectionMap(const std::shared_ptr<TCompressedConMap> cmpConnMap) {
    m_compressedConeConnectionMap = cmpConnMap;
  };

  std::pair<TStripConVec::const_iterator, TStripConVec::const_iterator> getConVec(uint32_t det,
                                                                                  unsigned char strip) const;

  std::pair<TCompressedConVec::const_iterator, TCompressedConVec::const_iterator> getCompConVec(uint32_t det) const;

  void setFirstTower(int tow) { m_firstTower = tow; };
  void setLastTower(int tow) { m_lastTower = tow; };

private:
  int m_firstTower;
  int m_lastTower;

  std::shared_ptr<TConMap> m_coneConnectionMap;
  std::shared_ptr<TCompressedConMap> m_compressedConeConnectionMap;

  COND_SERIALIZABLE;
};

#endif
