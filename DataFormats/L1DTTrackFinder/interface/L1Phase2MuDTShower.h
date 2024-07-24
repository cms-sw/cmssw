//-------------------------------------------------
//
//   Class L1Phase2MuDTShower
//
//   Description: shower primitive data for the
//                muon barrel Phase2 trigger
//
//
//   Author List: Carlos Vico  Oviedo Spain
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

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1Phase2MuDTShower {
public:
  //  Constructors
  L1Phase2MuDTShower();

  L1Phase2MuDTShower(int bx,  // BX estimation
                     int wh,  // Wheel
                     int sc,  // Sector
                     int st,  // Station
                     int ndigis, // Number of digis within shower
                     float avg_pos, // Averaged position of the shower
                     float avg_time); // Averaged time of the shower 
 

  virtual ~L1Phase2MuDTShower(){};

  // Operations
  int bxNum() const;

  int whNum() const;
  int scNum() const;
  int stNum() const;

  int ndigis() const;
  float avg_time() const;
  float avg_pos() const;
  
private:
  
  int m_bx;
  
  int m_wheel;
  int m_sector;
  int m_station;

  int m_ndigis;
  float m_avg_pos;
  float m_avg_time;

};

#endif
