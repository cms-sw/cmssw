#ifndef bLayer_H
#define bLayer_H

/** \class bLayer
 *  A layer of barrel volumes. Holds a list of volumes and 12 sectors.
 *  It is assumed that the geometry is 12-fold periodic in phi!
 *
 *  \author N. Amapane - INFN Torino
 */

#include "bSector.h"

class MagBLayer;

namespace magneticfield {

  class bLayer {
  public:
    /// Constructor from list of volumes
    bLayer(handles::const_iterator begin, handles::const_iterator end, bool debugFlag = false);

    /// Destructor
    ~bLayer() = default;

    /// Distance  from center along normal of sectors.
    const float RN() const { return theVolumes.front()->RN(); }

    /// Return the list of all volumes.
    const handles& volumes() const { return theVolumes; }

    /// Return sector at i (handling periodicity)
    //   const bSector & sector(int i) const;

    /// Min R (conservative guess).
    double minR() const;

    // Depends on volumeHandle::maxR(), which actually returns max RN.
    // (should be changed?)
    // double maxR() const;

    /// Construct the MagBLayer upon request.
    MagBLayer* buildMagBLayer() const;

  private:
    int size;  //< the number of volumes

    // Check periodicity;
    int bin(int i) const;

    std::vector<bSector> sectors;  // the sectors in this layer
    handles theVolumes;            // pointer to all volumes in this layer

    mutable MagBLayer* mlayer;
    const bool debug;
  };
}  // namespace magneticfield

#endif
