//-------------------------------------------------
//
//   Class L1MuDTChambPhDigi
//
//   Description: trigger primtive data for the
//                muon barrel Phase2 trigger
//
//
//   Author List:
//    Federica Primavera  Bologna INFN
//    Carlos Vico  Oviedo Spain,
//    Daniel Estrada Acevedo Oviedo Spain.
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
    : m_wheel(0),
      m_sector(0),
      m_station(0),
      m_superlayer(0),
      m_ndigis(0),
      m_bx(-100),
      m_min_wire(0),
      m_max_wire(0),
      m_avg_pos(0),
      m_avg_time(0) {
  m_wires_profile.resize(96, 0);
}

L1Phase2MuDTShower::L1Phase2MuDTShower(int wh,
                                       int sc,
                                       int st,
                                       int sl,
                                       int ndigis,
                                       int bx,
                                       int min_wire,
                                       int max_wire,
                                       float avg_pos,
                                       float avg_time,
                                       const std::vector<int> wires_profile)
    : m_wheel(wh),
      m_sector(sc),
      m_station(st),
      m_superlayer(sl),
      m_ndigis(ndigis),
      m_bx(bx),
      m_min_wire(min_wire),
      m_max_wire(max_wire),
      m_avg_pos(avg_pos),
      m_avg_time(avg_time),
      m_wires_profile(wires_profile) {}

//--------------
// Operations --
//--------------

int L1Phase2MuDTShower::whNum() const { return m_wheel; }

int L1Phase2MuDTShower::scNum() const { return m_sector; }

int L1Phase2MuDTShower::stNum() const { return m_station; }

int L1Phase2MuDTShower::slNum() const { return m_superlayer; }

int L1Phase2MuDTShower::ndigis() const { return m_ndigis; }

int L1Phase2MuDTShower::bxNum() const { return m_bx; }

int L1Phase2MuDTShower::minWire() const { return m_min_wire; }

int L1Phase2MuDTShower::maxWire() const { return m_max_wire; }

float L1Phase2MuDTShower::avg_time() const { return m_avg_time; }

float L1Phase2MuDTShower::avg_pos() const { return m_avg_pos; }

std::vector<int> L1Phase2MuDTShower::wiresProfile() const { return m_wires_profile; }
