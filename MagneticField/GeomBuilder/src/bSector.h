#ifndef bSector_H
#define bSector_H

/** \class bSector
 *  A sector of volumes in a barrel layer (i.e. only 1 element in R)
 *  One sector is composed of 1 or more rods.
 *
 *  \author N. Amapane - INFN Torino
 */

#include "bRod.h"

class MagBSector;

namespace magneticfield {

  class bSector {
  public:
    /// Default ctor is needed to have arrays.
    bSector();

    /// Constructor from list of volumes
    bSector(handles::const_iterator begin, handles::const_iterator end, bool debugVal = false);

    /// Destructor
    ~bSector() = default;

    /// Distance  from center along normal of sectors.
    const float RN() const { return volumes.front()->RN(); }

    /// Return all volumes in this sector
    const handles& getVolumes() const { return volumes; }

    /// Construct the MagBSector upon request.
    MagBSector* buildMagBSector() const;

  private:
    std::vector<bRod> rods;  // the rods in this layer
    handles volumes;         // pointers to all volumes in the sector
    mutable MagBSector* msector;
    bool debug;  // Allow assignment from other bSector objects
  };
}  // namespace magneticfield

#endif
