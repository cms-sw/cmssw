#ifndef bSlab_H
#define bSlab_H

/** \class MagGeoBuilderFromDDD::bSlab
 *  One or more slabs constitute a barrel rod.
 *  In most cases, a slab is a single volume, but in few cases it consists
 *  in several volumes contiguous in phi.
 *
 *  $Date: 2007/02/03 16:18:13 $
 *  $Revision: 1.3 $
 *  \author N. Amapane - INFN Torino
 */

#include "MagneticField/GeomBuilder/src/MagGeoBuilderFromDDD.h"
#include "MagneticField/GeomBuilder/src/volumeHandle.h"
#include "DataFormats/GeometryVector/interface/Pi.h"

class MagBSlab;

class MagGeoBuilderFromDDD::bSlab {
public:
  /// Constructor from list of volumes
  bSlab(handles::const_iterator begin, handles::const_iterator end);

  /// Destructor
  ~bSlab();
 
  /// Distance from center along sector normal.
  const float RN() const {
    return volumes.front()->RN();
  }

  /// Boundary in phi.
  // FIXME: use volumeHandle [max|min]Phi, which returns phi at median of
  // phi plane (not absolute limits). Used by: bRod ctor (only for dphi)
  Geom::Phi<float> minPhi() const;

  /// Boundary in phi.
  Geom::Phi<float> maxPhi() const;  

  /// Construct the MagBSlab upon request.
  MagBSlab* buildMagBSlab() const;

private:
  handles volumes; // pointers to all volumes in the slab
  mutable MagBSlab* mslab;
};

#endif
