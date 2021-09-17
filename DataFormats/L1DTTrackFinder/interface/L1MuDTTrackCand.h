//-------------------------------------------------
//
//   Class L1MuDTTrackCand
//
//   Description: output data for DTTF trigger
//
//
//   Author List: Jorge Troconiz  UAM Madrid
//
//
//--------------------------------------------------
#ifndef L1MuDTTrackCand_H
#define L1MuDTTrackCand_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//----------------------
// Base Class Headers --
//----------------------

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"

//---------------
// C++ Headers --
//---------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuDTTrackCand : public L1MuRegionalCand {
public:
  //  Constructors
  L1MuDTTrackCand();

  L1MuDTTrackCand(
      unsigned dataword, int bx, int uwh, int usc, int utag, int adr1, int adr2, int adr3, int adr4, int utc);

  L1MuDTTrackCand(unsigned type_idx,
                  unsigned phi,
                  unsigned eta,
                  unsigned pt,
                  unsigned charge,
                  unsigned ch_valid,
                  unsigned finehalo,
                  unsigned quality,
                  int bx,
                  int uwh,
                  int usc,
                  int utag,
                  int adr1,
                  int adr2,
                  int adr3,
                  int adr4);

  //  Destructor
  ~L1MuDTTrackCand() override;

  // Operations
  int whNum() const;
  int scNum() const;
  int stNum(int ust) const;
  int TCNum() const;
  int TrkTag() const;

  void setTC();
  void setAdd(int ust);

private:
  int wheel;
  int sector;
  int TrkTagCode;
  int TClassCode;
  int TrkAdd[4];
};

#endif
