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
#ifndef L1MuDTChambPhDigi_H
#define L1MuDTChambPhDigi_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//----------------------
// Base Class Headers --
//----------------------

//---------------
// C++ Headers --
//---------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuDTChambPhDigi {
public:
  //  Constructors
  L1MuDTChambPhDigi();

  L1MuDTChambPhDigi(
      int ubx, int uwh, int usc, int ust, int uphr, int uphb, int uqua, int utag, int ucnt, int urpc = -10);

  //  Destructor
  ~L1MuDTChambPhDigi();

  // Operations
  int bxNum() const;
  int whNum() const;
  int scNum() const;
  int stNum() const;
  int phi() const;
  int phiB() const;
  int code() const;
  int Ts2Tag() const;
  int BxCnt() const;
  int RpcBit() const;
  int UpDownTag() const;

private:
  int bx;
  int wheel;
  int sector;
  int station;
  int radialAngle;
  int bendingAngle;
  int qualityCode;
  int Ts2TagCode;
  int BxCntCode;
  int rpcBit;
};

#endif
