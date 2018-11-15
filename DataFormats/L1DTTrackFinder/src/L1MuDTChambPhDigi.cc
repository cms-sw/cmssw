//-------------------------------------------------
//
//   Class L1MuDTChambPhDigi
//
//   Description: input data for PHTF trigger
//
//
//   Author List: Jorge Troconiz  UAM Madrid
//
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------


//---------------
// C++ Headers --
//---------------

//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
L1MuDTChambPhDigi::L1MuDTChambPhDigi() {

  bx              = -100;
  wheel           = 0;
  sector          = 0;
  station         = 0;
  radialAngle     = 0;
  bendingAngle    = 0;
  qualityCode     = 7;
  Ts2TagCode      = 0;
  BxCntCode       = 0;
  rpcBit          = -10;
}

L1MuDTChambPhDigi::L1MuDTChambPhDigi( int ubx, int uwh, int usc, int ust,
                         int uphr, int uphb, int uqua, int utag, int ucnt, int urpc ) {

  bx              = ubx;
  wheel           = uwh;
  sector          = usc;
  station         = ust;
  radialAngle     = uphr;
  bendingAngle    = uphb;
  qualityCode     = uqua;
  Ts2TagCode      = utag;
  BxCntCode       = ucnt;
  rpcBit          = urpc;
}



//--------------
// Destructor --
//--------------
L1MuDTChambPhDigi::~L1MuDTChambPhDigi() {
}

//--------------
// Operations --
//--------------
int L1MuDTChambPhDigi::bxNum() const {
  return bx;
}

int L1MuDTChambPhDigi::whNum() const {
  return wheel;
}
int L1MuDTChambPhDigi::scNum() const {
  return sector;
}
int L1MuDTChambPhDigi::stNum() const {
  return station;
}

int L1MuDTChambPhDigi::phi() const {
  return radialAngle;
}

int L1MuDTChambPhDigi::phiB() const {
  return bendingAngle;
}

int L1MuDTChambPhDigi::code() const {
  return qualityCode;
}

int L1MuDTChambPhDigi::Ts2Tag() const {
  return Ts2TagCode%2;
}

int L1MuDTChambPhDigi::BxCnt() const {
  return BxCntCode;
}

int L1MuDTChambPhDigi::RpcBit() const {
  return rpcBit;
}

int L1MuDTChambPhDigi::UpDownTag()	const{
  return Ts2TagCode/2 ;
}
