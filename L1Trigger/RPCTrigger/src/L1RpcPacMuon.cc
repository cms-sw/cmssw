/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2002                                                      *
*                                                                              *
*******************************************************************************/
#include "L1Trigger/RPCTrigger/src/L1RpcPacMuon.h"

///Default constructor. No muon.
L1RpcPacMuon::L1RpcPacMuon(): L1RpcMuon() {}

///Constructor.
L1RpcPacMuon::L1RpcPacMuon(const L1RpcPattern& pattern, int quality, unsigned short firedPlanes):
    L1RpcMuon(pattern.GetCode(), quality, pattern.GetSign(), pattern.GetNumber(), firedPlanes) 
    { }

void L1RpcPacMuon::SetAll(const L1RpcPattern& pattern, int quality, unsigned short firedPlanes) {
  PatternNum = pattern.GetNumber();
  PtCode = pattern.GetCode();
  Sign = pattern.GetSign();
  Quality = quality;
  FiredPlanes = firedPlanes;
}

void L1RpcPacMuon::SetPatternNum(int patternNum) {
  PatternNum = patternNum;
}

bool L1RpcPacMuon::operator < (const L1RpcPacMuon& pacMuon) const {
  if( this->Quality < pacMuon.Quality )
    return true;
  else if( this->Quality > pacMuon.Quality )
    return false;
  else { //==
    if( this->PtCode < pacMuon.PtCode )
      return true;
    else if( this->PtCode > pacMuon.PtCode )
      return false;
    else { //==
      if( this->Sign < pacMuon.Sign )
        return true;
      else
        return false;
    }
  }
}

bool L1RpcPacMuon::operator > (const L1RpcPacMuon& pacMuon) const {
  if( this->Quality > pacMuon.Quality )
    return true;
  else if( this->Quality < pacMuon.Quality )
    return false;
  else { //==
    if( this->PtCode > pacMuon.PtCode )
      return true;
    else if( this->PtCode < pacMuon.PtCode )
      return false;
    else { //==
      if( this->Sign > pacMuon.Sign )
        return true;
      else
        return false;
    }
  }
}
bool L1RpcPacMuon::operator == (const L1RpcPacMuon& pacMuon) const {
  if( this->Quality == pacMuon.Quality &&
      this->PtCode == pacMuon.PtCode &&
      this->Sign == pacMuon.Sign          )
    return true;
  else
    return false;
}
