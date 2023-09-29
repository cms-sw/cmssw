/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*   Maciej Wróbel (wroblisko@gmail.com)
*   Jan Kašpar (jan.kaspar@cern.ch)
*
****************************************************************************/

#ifndef CondFormats_PPSObjects_TotemAnalysisMask
#define CondFormats_PPSObjects_TotemAnalysisMask

#include "CondFormats/PPSObjects/interface/TotemSymbId.h"
#include "CondFormats/Serialization/interface/Serializable.h"
#include <map>
#include <set>

//----------------------------------------------------------------------------------------------------

/**
 *\brief Contains data on masked channels of a VFAT.
 */
class TotemVFATAnalysisMask {
public:
  TotemVFATAnalysisMask() : fullMask(false) {}

  /// whether all channels of the VFAT shall be masked
  bool fullMask;

  /// list of channels to be masked
  std::set<unsigned char> maskedChannels;

  COND_SERIALIZABLE;
};

//----------------------------------------------------------------------------------------------------

/**
 *\brief Channel-mask mapping.
 **/
class TotemAnalysisMask {
public:
  std::map<TotemSymbID, TotemVFATAnalysisMask> analysisMask;

  void insert(const TotemSymbID& sid, const TotemVFATAnalysisMask& vam);
  void print(std::ostream& os) const;

  COND_SERIALIZABLE;
};

std::ostream& operator<<(std::ostream& os, TotemAnalysisMask mask);

#endif