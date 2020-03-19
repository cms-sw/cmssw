#ifndef bRod_H
#define bRod_H

/** \class MagGeoBuilderFromDDD::bRod
 *  A rod of volumes in a barrel sector.
 *  A rod is made of several "slabs".
 *
 *  \author N. Amapane - INFN Torino
 */

#include "bSlab.h"

class MagBRod;

namespace magneticfield {

  class bRod {
  public:
    /// Constructor from list of volumes
    bRod(handles::const_iterator begin, handles::const_iterator end, bool debugVal = false);

    /// Destructor
    ~bRod() = default;

    /// Distance from center along sector normal.
    const float RN() const { return volumes.front()->RN(); }

    /// Construct the MagBRod upon request.
    MagBRod* buildMagBRod() const;

  private:
    std::vector<bSlab> slabs;
    handles volumes;  // pointers to all volumes in the rod
    mutable MagBRod* mrod;
    bool debug;  // Allow assignment from other bRod objects
  };
}  // namespace magneticfield

#endif
