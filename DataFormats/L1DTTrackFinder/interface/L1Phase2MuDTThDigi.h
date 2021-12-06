#ifndef L1Phase2MuDTThDigi_H
#define L1Phase2MuDTThDigi_H

class L1Phase2MuDTThDigi {
public:
  //  Constructors
  L1Phase2MuDTThDigi();

  L1Phase2MuDTThDigi(int bx, int wh, int sc, int st, int z, int k, int qual, int idx, int t0, int chi2, int rpc = -10);

  virtual ~L1Phase2MuDTThDigi(){};

  // Operations
  int bxNum() const;

  int whNum() const;
  int scNum() const;
  int stNum() const;

  int z() const;
  int k() const;

  int quality() const;
  int index() const;

  int t0() const;
  int chi2() const;

  int rpcFlag() const;

private:
  int m_bx;
  int m_wheel;
  int m_sector;
  int m_station;

  int m_zGlobal;
  int m_kSlope;

  int m_qualityCode;
  int m_index;

  int m_t0;
  int m_chi2;

  int m_rpcFlag;
};

#endif
