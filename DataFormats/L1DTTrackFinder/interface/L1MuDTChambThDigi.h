//-------------------------------------------------
//
//   Class L1MuDTChambThDigi
//
//   Description: input data for ETTF trigger
//
//
//   Author List: Jorge Troconiz  UAM Madrid
//
//
//--------------------------------------------------
#ifndef L1MuDTChambThDigi_H
#define L1MuDTChambThDigi_H

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

typedef unsigned char myint8;

class L1MuDTChambThDigi {
public:
  //  Constructors
  L1MuDTChambThDigi();

  L1MuDTChambThDigi(int ubx, int uwh, int usc, int ust, int* uos, int* uqual);

  L1MuDTChambThDigi(int ubx, int uwh, int usc, int ust, int* uos);

  //  Destructor
  ~L1MuDTChambThDigi();

  // Operations
  int bxNum() const;
  int whNum() const;
  int scNum() const;
  int stNum() const;

  int code(const int i) const;
  int position(const int i) const;
  int quality(const int i) const;

private:
  int bx;
  int wheel;
  int sector;
  int station;

  myint8 m_outPos[7];
  myint8 m_outQual[7];
};

#endif
