/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2002                                                      *
*                                                                              *
*******************************************************************************/
#include "L1Trigger/RPCTrigger/src/L1RpcPacMuon.h"

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
