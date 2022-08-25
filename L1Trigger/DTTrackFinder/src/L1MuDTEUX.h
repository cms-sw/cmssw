//-------------------------------------------------
//
/**  \class L1MuDTEUX
 *
 *   Extrapolator:
 *
 *   The Extrapolator receives a start and 
 *   a target track segment and performs the 
 *   actual extrapolation
 *
 *
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUDT_EUX_H
#define L1MUDT_EUX_H

//---------------
// C++ Headers --
//---------------

#include <utility>
#include <map>

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
class L1MuDTTrackSegPhi;
class L1MuDTSectorProcessor;
class L1MuDTSEU;
class L1MuDTExtLut;
class L1MuDTTFParameters;
class L1MuDTExtLut;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuDTEUX {
public:
  /// constructor
  L1MuDTEUX(const L1MuDTSectorProcessor& sp, const L1MuDTSEU& seu, int id);

  /// destructor
  ~L1MuDTEUX();

  /// equal operator
  bool operator==(const L1MuDTEUX&) const;

  /// run Extrapolator
  void run(const L1MuDTExtLut& extLUTs, const L1MuDTTFParameters& pars);

  /// reset Extrapolator
  void reset();

  /// load data into EUX
  void load(const L1MuDTTrackSegPhi* start_ts, const L1MuDTTrackSegPhi* target_ts);

  /// return pointer to start and target track segment
  std::pair<const L1MuDTTrackSegPhi*, const L1MuDTTrackSegPhi*> ts() const;

  /// helper class for finding the best and second best extrapolation
  class EUX_Comp {
  public:
    EUX_Comp(const L1MuDTEUX* k = nullptr) : _not(k) {}
    bool operator()(const L1MuDTEUX* first, const L1MuDTEUX* second) const {
      if (!second->result())
        return false;
      if (_not != nullptr && *first == *_not)
        return true;
      if (_not != nullptr && *second == *_not)
        return false;
      return (first->quality() < second->quality());
    }

  private:
    const L1MuDTEUX* _not;
  };

  /// return Extrapolation identifier
  inline int id() const { return m_id; }

  /// return extrapolation result
  inline bool result() const { return m_result; }

  /// return extrapolation quality
  inline unsigned int quality() const { return m_quality; }

  /// return extrapolation address (0-11) (address = 15 indicates negative ext. result)
  inline unsigned short int address() const { return m_address; }

private:
  /// output sector numbers in the range -6 to +5
  int sec_mod(int) const;

private:
  const L1MuDTSectorProcessor& m_sp;
  const L1MuDTSEU& m_seu;  // reference to Single Extrapolation Unit
  int m_id;                // index of start TS

  bool m_result;                 //@@ 1 bit
  unsigned short int m_quality;  //@@ 1 bit
  unsigned short int m_address;  //@@ 4 bits

  const L1MuDTTrackSegPhi* m_start;   // start track segment
  const L1MuDTTrackSegPhi* m_target;  // target track segment

  int const theExtFilter = 1;           // extrapolation quality filter
  unsigned short const nbit_phi = 12;   // number of bits used for phi
  unsigned short const nbit_phib = 10;  // number of bits used for phib
};
#endif
