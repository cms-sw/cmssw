/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*   Maciej Wróbel (wroblisko@gmail.com)
*   Jan Kašpar (jan.kaspar@cern.ch)
*
****************************************************************************/

#ifndef CondFormats_PPSObjects_TotemDAQMapping
#define CondFormats_PPSObjects_TotemDAQMapping

#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/PPSObjects/interface/TotemFramePosition.h"
#include "CondFormats/PPSObjects/interface/TotemT2FramePosition.h"

#include "CondFormats/PPSObjects/interface/TotemSymbId.h"

#include <map>

//----------------------------------------------------------------------------------------------------

/**
 *\brief Contains mappind data related to a VFAT.
 */
class TotemVFATInfo {
public:
  /// the symbolic id
  TotemSymbID symbolicID;

  /// the hardware ID (16 bit)
  unsigned int hwID;

  friend std::ostream& operator<<(std::ostream& s, const TotemVFATInfo& fp);

  COND_SERIALIZABLE;
};

//----------------------------------------------------------------------------------------------------

/**
 *\brief The mapping between FramePosition and VFATInfo.
 */
class TotemDAQMapping {
public:
  std::map<TotemFramePosition, TotemVFATInfo> VFATMapping;

  /// Hw Id mapping for Totem Timing (dynamical mapping in Sampic)
  struct TotemTimingPlaneChannelPair {
    int plane;
    int channel;

    TotemTimingPlaneChannelPair(const int& plane = -1, const int& channel = -1) : plane(plane), channel(channel){};
    COND_SERIALIZABLE;
  };
  std::map<uint8_t, TotemTimingPlaneChannelPair> totemTimingChannelMap;

  void insert(const TotemFramePosition& fp, const TotemVFATInfo& vi);
  void insert(const TotemT2FramePosition& fp2, const TotemVFATInfo& vi);

  /// Given the hardware ID, returns the corresponding Plane, Channel pair (TotemTimingPlaneChannelPair)
  const TotemTimingPlaneChannelPair getTimingChannel(const uint8_t hwId) const;

  COND_SERIALIZABLE;
};

#endif
