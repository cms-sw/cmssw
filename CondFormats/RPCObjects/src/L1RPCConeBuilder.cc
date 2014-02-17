// -*- C++ -*-
//
// Package:     RPCObjects
// Class  :     L1RPCConeBuilder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Tomasz Frueboes
//         Created:  Fri Feb 22 12:26:49 CET 2008
// $Id: L1RPCConeBuilder.cc,v 1.6 2009/04/10 15:36:30 fruboes Exp $
//

#include "CondFormats/RPCObjects/interface/L1RPCConeBuilder.h"



//
L1RPCConeBuilder::L1RPCConeBuilder() :
      m_firstTower(0), 
      m_lastTower(-1)
{

}

// L1RPCConeBuilder::L1RPCConeBuilder(const L1RPCConeBuilder& rhs)
// {
//    // do actual copying here;
// }

L1RPCConeBuilder::~L1RPCConeBuilder()
{
}

std::pair<L1RPCConeBuilder::TStripConVec::const_iterator, L1RPCConeBuilder::TStripConVec::const_iterator> 
    L1RPCConeBuilder::getConVec(uint32_t det, unsigned char strip) const
{

  L1RPCConeBuilder::TStripConVec::const_iterator itBeg = L1RPCConeBuilder::TStripConVec().end();
  L1RPCConeBuilder::TStripConVec::const_iterator itEnd = itBeg;
  
  TConMap::const_iterator it1 = m_coneConnectionMap->find(det);
  if (it1 != m_coneConnectionMap->end()){
    TStrip2ConVec::const_iterator it2 = it1->second.find(strip);
    if (it2 != it1->second.end()){
      itBeg = it2->second.begin();
      itEnd = it2->second.end();
    }
  } 
  
  return std::make_pair(itBeg,itEnd);
}

std::pair<L1RPCConeBuilder::TCompressedConVec::const_iterator, L1RPCConeBuilder::TCompressedConVec::const_iterator> 
    L1RPCConeBuilder::getCompConVec(uint32_t det) const 
{
  L1RPCConeBuilder::TCompressedConVec::const_iterator itBeg = L1RPCConeBuilder::TCompressedConVec().end();
  L1RPCConeBuilder::TCompressedConVec::const_iterator itEnd = itBeg;
  
  if(m_compressedConeConnectionMap->find(det)!=m_compressedConeConnectionMap->end()){
    itBeg =  m_compressedConeConnectionMap->find(det)->second.begin();
    itEnd =  m_compressedConeConnectionMap->find(det)->second.end();
  }
  
  return std::make_pair(itBeg,itEnd);
}

