#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTThDigi.h"

L1Phase2MuDTThDigi::L1Phase2MuDTThDigi()
    : m_bx(-100),
      m_wheel(0),
      m_sector(0),
      m_station(0),
      m_zGlobal(0),
      m_kSlope(0),
      m_qualityCode(-1),
      m_index(0),
      m_t0(0),
      m_chi2(0),
      m_rpcFlag(-10) {}

L1Phase2MuDTThDigi::L1Phase2MuDTThDigi(
    int bx, int wh, int sc, int st, int z, int k, int qual, int idx, int t0, int chi2, int rpc)
    : m_bx(bx),
      m_wheel(wh),
      m_sector(sc),
      m_station(st),
      m_zGlobal(z),
      m_kSlope(k),
      m_qualityCode(qual),
      m_index(idx),
      m_t0(t0),
      m_chi2(chi2),
      m_rpcFlag(rpc) {}

int L1Phase2MuDTThDigi::bxNum() const { return m_bx; }

int L1Phase2MuDTThDigi::whNum() const { return m_wheel; }

int L1Phase2MuDTThDigi::scNum() const { return m_sector; }

int L1Phase2MuDTThDigi::stNum() const { return m_station; }

int L1Phase2MuDTThDigi::z() const { return m_zGlobal; }

int L1Phase2MuDTThDigi::k() const { return m_kSlope; }

int L1Phase2MuDTThDigi::quality() const { return m_qualityCode; }

int L1Phase2MuDTThDigi::index() const { return m_index; }

int L1Phase2MuDTThDigi::t0() const { return m_t0; }

int L1Phase2MuDTThDigi::chi2() const { return m_chi2; }

int L1Phase2MuDTThDigi::rpcFlag() const { return m_rpcFlag; }
