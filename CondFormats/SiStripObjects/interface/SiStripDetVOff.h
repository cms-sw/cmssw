#ifndef SiStripDetVOff_h
#define SiStripDetVOff_h

#include<vector>
#include<map>
#include<iostream>
#include<boost/cstdint.hpp>

/**
 * This class stores the information if the HV or LV of a detId were off. <br>
 * Internally it uses two bits to store the information about LV and HV. It saves a uint32_t
 * containing the detId number and these additional bits, which are stored in the first
 * position. This is realized by the put method using a bit shift so that the actual
 * number written in the database is: detId|HV|LV. <br>
 * The getDetIds method builds and returns a vector with detIds, removing the additional bits.
 * It has three methods to extract the information: <br>
 * - IsModuleVOff returning the true if any of HV or LV is off
 * - IsModuleLVOff/IsModuleHVOff returning true if the corresponding value is off.
 */

class SiStripDetVOff
{
 public:

  typedef std::vector<uint32_t>::iterator       vOffIterator;
  typedef std::vector<uint32_t>::const_iterator constVoffIterator;
  typedef std::vector<bool>::const_iterator     constVboolIterator;

  // Bitmasks used to retrieve LV and HV information
  static const short LVmask = 0x1;  // <--- 01
  static const short HVmask = 0x2;  // <--- 10
  static const unsigned int eightBitMask = 0xFFFFFFFF;
  static const short bitShift = 2;

  SiStripDetVOff(){};
  ~SiStripDetVOff(){};

  /// Insert information for a single detId
  bool put(const uint32_t DetId, const bool LVoff, const bool HVoff);

  /// Insert information for a vector of detIds
  bool put(std::vector<uint32_t>& DetId, std::vector<bool>& HVoff, std::vector<bool>& LVoff);

  bool operator == (const SiStripDetVOff& d) const { return d.v_Voff==v_Voff; } 

  void getDetIds(std::vector<uint32_t>& DetIds_) const;

  /// Returns true if either LV or HV are off
  bool IsModuleVOff(const uint32_t DetID) const;

  bool IsModuleHVOff(const uint32_t DetID) const;

  bool IsModuleLVOff(const uint32_t DetID) const;

  void printDebug(std::stringstream & ss) const;
  void printSummary(std::stringstream & ss) const;

 private:

  std::vector<uint32_t> v_Voff; 
};

#endif
