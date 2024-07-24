//-------------------------------------------------
//
//   Class L1MuDTChambPhDigi
//
//   Description: trigger primtive data for the
//                muon barrel Phase2 trigger
//
//
//   Author List: Federica Primavera  Bologna INFN
//
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTShower.h"

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
L1Phase2MuDTShower::L1Phase2MuDTShower()
    : m_bx(-100),
      m_wheel(0),
      m_sector(0),
      m_station(0),
      m_ndigis(0),
      m_avg_pos(0),
      m_avg_time(0) {}

L1Phase2MuDTShower::L1Phase2MuDTShower(
    int bx, int wh, int sc, int st, int ndigis, float avg_pos, float avg_time)
    : m_bx(bx),
      m_wheel(wh),
      m_sector(sc),
      m_station(st),
      m_ndigis(ndigis),
      m_avg_pos(avg_pos),
      m_avg_time(avg_time) {}

//--------------
// Operations --
//--------------

int L1Phase2MuDTShower::whNum() const { return m_wheel; }

int L1Phase2MuDTShower::scNum() const { return m_sector; }

int L1Phase2MuDTShower::stNum() const { return m_station; }

int L1Phase2MuDTShower::bxNum() const { return m_bx; }

int L1Phase2MuDTShower::ndigis() const { return m_ndigis; }

float L1Phase2MuDTShower::avg_time() const { return m_avg_time; }

float L1Phase2MuDTShower::avg_pos() const { return m_avg_pos; }
