#ifndef bLayer_H
#define bLayer_H

/** \class MagGeoBuilderFromDDD::bLayer
 *  A layer of barrel volumes. Holds a list of volumes and 12 sectors.
 *  It is assumed that the geometry is 12-fold periodic in phi!
 *
 *  $Date: 2005/09/27 15:15:52 $
 *  $Revision: 1.2 $
 *  \author N. Amapane - INFN Torino
 */

#include "MagneticField/GeomBuilder/src/MagGeoBuilderFromDDD.h"
#include "MagneticField/GeomBuilder/src/volumeHandle.h"
#include "MagneticField/GeomBuilder/src/bSector.h"

class MagBLayer;

class MagGeoBuilderFromDDD::bLayer {
public:
  /// Constructor from list of volumes
  bLayer(handles::const_iterator begin, handles::const_iterator end);

  /// Destructor
  ~bLayer();

  /// Distance  from center along normal of sectors.
  const float RN() const {
    return theVolumes.front()->RN();
  }

  /// Return the list of all volumes.
  const handles & volumes() const {return theVolumes;}

  /// Return sector at i (handling periodicity)
  //   const bSector & sector(int i) const;

  /// Min R (conservative guess).
  double minR() const;

  // Depends on volumeHandle::maxR(), which actually returns max RN.
  // (should be changed?)
  // double maxR() const;

  /// Construct the MagBLayer upon request.
  MagBLayer * buildMagBLayer() const;

private:
  int size; //< the number of volumes

  // Check periodicity;
  int bin(int i) const;

  std::vector<bSector> sectors; // the sectors in this layer
  handles theVolumes;  // pointer to all volumes in this layer

  mutable MagBLayer * mlayer;
};
#endif

