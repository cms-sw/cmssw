#ifndef L1CSCSPStatusDigi_h
#define L1CSCSPStatusDigi_h

#include <cstring>

class CSCTFUnpacker;

class L1CSCSPStatusDigi {
private:
  unsigned short sp_slot;     //
  unsigned short l1a_bxn;     // Take from the SP header
  unsigned short fmm_status;  // Take from the SP header

  unsigned short se;  // Make logical OR of all tbins for each of 15(endcap) SE bits
  unsigned short sm;  // Make logical OR of all tbins for each of 15(endcap) SM bits
  unsigned long bx;   // Make logical OR of all tbins for each of 15(endcap)+2(barrel) BX bits
  unsigned long af;   // Make logical OR of all tbins for each of 15(endcap)+2(barrel) AF bits
  unsigned long vp;   // Make logical OR of all tbins for each of 15(endcap)+2(barrel) VP/VQ bits
  enum {
    IDLE = 1,
    CARRIER = 2,
    NORMAL = 4,
    ERROR = 8,
    FIFO = 16
  };                               // States of optical receivers + Alignment FIFO full OR empty status
  unsigned short link_status[15];  // Optical receiver status ORed for all tbins
  unsigned short mpc_link_id;      // MPC_id and link# from MEx Data Record ORed for all tbins

  unsigned long track_cnt;
  unsigned long orbit_cnt;

  friend class CSCTFUnpacker;

public:
  unsigned short slot(void) const throw() { return sp_slot; }
  unsigned short BXN(void) const throw() { return l1a_bxn; }
  unsigned short FMM(void) const throw() { return fmm_status; }
  unsigned short SEs(void) const throw() { return se; }
  unsigned short SMs(void) const throw() { return sm; }
  unsigned long BXs(void) const throw() { return bx; }
  unsigned long AFs(void) const throw() { return af; }
  unsigned long VPs(void) const throw() { return vp; }
  unsigned short link(int link) const throw() { return link_status[link]; }

  unsigned long track_counter(void) const throw() { return track_cnt; }
  unsigned long orbit_counter(void) const throw() { return orbit_cnt; }

  L1CSCSPStatusDigi(void) { bzero(this, sizeof(L1CSCSPStatusDigi)); }
  ~L1CSCSPStatusDigi(void) {}
};

#endif
