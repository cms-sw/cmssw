#ifndef eLayer_H
#define eLayer_H

/** \class MagGeoBuilderFromDDD::eLayer
 *  A layer of volumes in an endcap sector.
 *
 *  $Date: 2005/09/27 15:15:52 $
 *  $Revision: 1.2 $
 *  \author N. Amapane - INFN Torino
 */

#include "MagneticField/GeomBuilder/src/MagGeoBuilderFromDDD.h"
#include "MagneticField/GeomBuilder/src/volumeHandle.h"
#include "MagneticField/GeomBuilder/src/bSector.h"

class MagELayer;

class MagGeoBuilderFromDDD::eLayer {
public:
  /// Constructor from list of volumes
  eLayer(handles::const_iterator begin, handles::const_iterator end);

  /// Destructor
  ~eLayer();

//   /// Return the list of all volumes.
//   const handles & volumes() const {return theVolumes;}

  /// Construct the MagELayer upon request.
  MagELayer * buildMagELayer() const;

private:
  handles theVolumes;  // pointer to all volumes in this layer
  mutable MagELayer * mlayer;
};
#endif

