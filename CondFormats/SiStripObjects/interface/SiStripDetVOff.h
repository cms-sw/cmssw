#ifndef SiStripDetVOff_h
#define SiStripDetVOff_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include<vector>
#include<map>
#include<iostream>
#include<boost/cstdint.hpp>
#include <string>

class TrackerTopology;

/**
 * This class stores the information if the HV or LV of a detId were off. <br>
 * Internally it uses two bits to store the information about HV and LV. It saves a uint32_t
 * containing the detId number and these additional bits, which are stored in the first
 * position. This is realized by the put method using a bit shift so that the actual
 * number written in the database is: detId|HV|LV. <br>
 * The getDetIds method builds and returns a vector with detIds, removing the additional bits.
 * It has three methods to extract the information: <br>
 * - IsModuleVOff returning the true if any of HV or LV is off
 * - IsModuleHVOff/IsModuleLVOff returning true if the corresponding value is off.
 *
 * The printSummary method uses SiStripDetSummary to print both LV off and HV off summaries.
 * See description of the SiStripDetSummary class therein. <br>
 * The printDebug method prints the status of HV and LV for all DetIds that have at least
 * one of the two off.
 */

class SiStripDetVOff
{
 public:

  typedef std::vector<uint32_t>::iterator       vOffIterator;
  typedef std::vector<uint32_t>::const_iterator constVoffIterator;
  typedef std::vector<int>::const_iterator     constVboolIterator;

  // Bitmasks used to retrieve LV and HV information
  static const short LVmask = 0x1;    // <--- 01
  static const unsigned int LVonMask = 0xFFFFFFFE;    // <--- the last 4 bits are 1110. All the other bits are 1.
  static const short HVmask = 0x2;    // <--- 10
  static const unsigned int HVonMask = 0xFFFFFFFD;    // <--- the last 4 bits are 1101. All the other bits are 1.
  static const unsigned int allOnMask = 0x03;   // <--- 2 bits are 11.
  static const unsigned int eightBitMask = 0xFFFFFFFF;
  static const short bitShift = 2;

  SiStripDetVOff() {}
  ~SiStripDetVOff() {}
  SiStripDetVOff( const SiStripDetVOff & toCopy ) { toCopy.getVoff(v_Voff); }

  /// Needed by the copy constructor
  void getVoff(std::vector<uint32_t>& vOff_) const { vOff_ = v_Voff; }

  /// Insert information for a single detId
  bool put(const uint32_t DetId, const int HVoff, const int LVoff);

  /// Insert information for a vector of detIds
  bool put(std::vector<uint32_t>& DetId, std::vector<int>& HVoff, std::vector<int>& LVoff);

  bool operator == (const SiStripDetVOff& d) const { return d.v_Voff==v_Voff; } 

  void getDetIds(std::vector<uint32_t>& DetIds_) const;

  /// Returns true if either HV or LV are off
  bool IsModuleVOff(const uint32_t DetID) const;

  bool IsModuleHVOff(const uint32_t DetID) const;

  bool IsModuleLVOff(const uint32_t DetID) const;

  void printDebug(std::stringstream & ss, const TrackerTopology*) const;
  void printSummary(std::stringstream & ss, const TrackerTopology*) const;

  /// Returns the total number of modules with LV off
  int getLVoffCounts() const;
  /// Returns the total number of modules with HV off
  int getHVoffCounts() const;

  /// Changes the bits in the stored value according to on/off voltages
  void setBits( uint32_t & enDetId, const int HVoff, const int LVoff );

 private:

  std::vector<uint32_t> v_Voff; 

 COND_SERIALIZABLE;
};

#endif
