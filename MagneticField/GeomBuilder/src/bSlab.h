#ifndef bSlab_H
#define bSlab_H

/** \class MagGeoBuilderFromDDD::bSlab
 *  One or more slabs constitute a barrel rod.
 *  In most cases, a slab is a single volume, but in few cases it consists
 *  in several volumes contiguous in phi.
 *
 *  \author N. Amapane - INFN Torino
 */

#include "BaseVolumeHandle.h"

#include "DataFormats/GeometryVector/interface/Phi.h"

#include "MagneticField/VolumeGeometry/interface/MagVolume6Faces.h"
#include "MagneticField/Layers/interface/MagBSlab.h"
#include "MagneticField/Layers/interface/MagVerbosity.h"

class MagBSlab;

namespace magneticfield {

  class bSlab {
  public:
    /// Constructor from list of volumes
    bSlab(handles::const_iterator startIter, handles::const_iterator endIter, bool debugVal = false);

    /// Destructor
    ~bSlab() = default;

    /// Distance from center along sector normal.
    const float RN() const { return volumes.front()->RN(); }

    /// Boundary in phi.
    // FIXME: use volumeHandle [max|min]Phi, which returns phi at median of
    // phi plane (not absolute limits). Used by: bRod ctor (only for dphi)
    Geom::Phi<float> minPhi() const;

    /// Boundary in phi.
    Geom::Phi<float> maxPhi() const;

    /// Construct the MagBSlab upon request.
    MagBSlab* buildMagBSlab() const;

  private:
    handles volumes;  // pointers to all volumes in the slab
    mutable MagBSlab* mslab;
    bool debug;  // Allow assignment
  };
}  // namespace magneticfield

#endif
