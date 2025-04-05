//-------------------------------------------------
//
//   Class L1Phase2MuDTShower
//
//   Description: shower primitive data for the
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
#ifndef L1Phase2MuDTShower_H
#define L1Phase2MuDTShower_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//----------------------
// Base Class Headers --
//----------------------

//---------------
// C++ Headers --
//---------------
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1Phase2MuDTShower {
public:
  //  Constructors
  L1Phase2MuDTShower();

  L1Phase2MuDTShower(int wh,                               // Wheel
                     int sc,                               // Sector
                     int st,                               // Station
                     int sl,                               // Superlayer
                     int ndigis,                           // Number of digis within shower
                     int bx,                               // BX estimation
                     int min_wire,                         // Minimum wire
                     int max_wire,                         // Maximum wire
                     float avg_pos,                        // Averaged position of the shower
                     float avg_time,                       // Averaged time of the shower
                     const std::vector<int> wires_profile  // Wires profile
  );

  virtual ~L1Phase2MuDTShower() {};

  // Operations

  int whNum() const;
  int scNum() const;
  int stNum() const;
  int slNum() const;
  int ndigis() const;
  int bxNum() const;
  int minWire() const;
  int maxWire() const;
  float avg_time() const;
  float avg_pos() const;
  std::vector<int> wiresProfile() const;

private:
  int m_wheel;
  int m_sector;
  int m_station;
  int m_superlayer;
  int m_ndigis;
  int m_bx;
  int m_min_wire;
  int m_max_wire;
  float m_avg_pos;
  float m_avg_time;
  std::vector<int> m_wires_profile;
};

#endif
