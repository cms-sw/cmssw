/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2002                                                      *
*                                                                              *
*******************************************************************************/
#include "L1Trigger/RPCTrigger/interface/RPCPacMuon.h"

///Default constructor. No muon.
RPCPacMuon::RPCPacMuon(): RPCMuon() {}

///Constructor.
RPCPacMuon::RPCPacMuon(const RPCPattern& pattern, int quality, unsigned short firedPlanes):
    RPCMuon(pattern.getCode(), quality, pattern.getSign(), pattern.getNumber(), firedPlanes) 
    { }

void RPCPacMuon::setAll(const RPCPattern& pattern, int quality, unsigned short firedPlanes) {
  m_PatternNum = pattern.getNumber();
  m_PtCode = pattern.getCode();
  m_Sign = pattern.getSign();
  m_Quality = quality;
  m_FiredPlanes = firedPlanes;
}

void RPCPacMuon::setPatternNum(int patternNum) {
  m_PatternNum = patternNum;
}

bool RPCPacMuon::operator <(const RPCPacMuon& pacMuon) const {
  if( this->m_Quality < pacMuon.m_Quality)
    return true;
  else if( this->m_Quality > pacMuon.m_Quality)
    return false;
  else { //==
    if( this->m_PtCode < pacMuon.m_PtCode)
      return true;
    else if( this->m_PtCode > pacMuon.m_PtCode)
      return false;
    else { //==
      //if( this->m_Sign < pacMuon.m_Sign)
      if( this->m_Sign > pacMuon.m_Sign)
        return true;
      else
        return false;
    }
  }
}

bool RPCPacMuon::operator >(const RPCPacMuon& pacMuon) const {
  if( this->m_Quality > pacMuon.m_Quality)
    return true;
  else if( this->m_Quality < pacMuon.m_Quality)
    return false;
  else { //==
    if( this->m_PtCode > pacMuon.m_PtCode)
      return true;
    else if( this->m_PtCode < pacMuon.m_PtCode)
      return false;
    else { //==
      if( this->m_Sign < pacMuon.m_Sign)
        return true;
      else
        return false;
    }
  }
}
bool RPCPacMuon::operator ==(const RPCPacMuon& pacMuon) const {
  if( this->m_Quality == pacMuon.m_Quality &&
      this->m_PtCode == pacMuon.m_PtCode &&
      this->m_Sign == pacMuon.m_Sign)
    return true;
  else
    return false;
}
