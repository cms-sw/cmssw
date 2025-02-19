#ifndef bSector_H
#define bSector_H

/** \class MagGeoBuilderFromDDD::bSector
 *  A sector of volumes in a barrel layer (i.e. only 1 element in R)
 *  One sector is composed of 1 or more rods.
 *
 *  $Date: 2005/09/27 15:15:52 $
 *  $Revision: 1.2 $
 *  \author N. Amapane - INFN Torino
 */

#include "MagneticField/GeomBuilder/src/MagGeoBuilderFromDDD.h"
#include "MagneticField/GeomBuilder/src/volumeHandle.h"
#include "MagneticField/GeomBuilder/src/bRod.h"

class MagBSector;

class MagGeoBuilderFromDDD::bSector {
public:
  /// Default ctor is needed to have arrays.
  bSector();

  /// Constructor from list of volumes
  bSector(handles::const_iterator begin, handles::const_iterator end);

  /// Destructor
  ~bSector();

  /// Distance  from center along normal of sectors.
  const float RN() const {
    return volumes.front()->RN();
  }

  /// Return all volumes in this sector
  const handles & getVolumes() const {return volumes;}

  /// Construct the MagBSector upon request.
  MagBSector* buildMagBSector() const;

private:
  std::vector<bRod> rods; // the rods in this layer
  handles volumes;   // pointers to all volumes in the sector
  mutable MagBSector* msector;
};
#endif
