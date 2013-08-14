#ifndef eSector_H
#define eSector_H

/** \class MagGeoBuilderFromDDD::eSector
 *  A sector of volumes in the endcap.
 *  One sector is composed of several layers (eLayer)
 *
 *  $Date: 2005/09/27 15:15:52 $
 *  $Revision: 1.2 $
 *  \author N. Amapane - INFN Torino
 */

#include "MagneticField/GeomBuilder/src/MagGeoBuilderFromDDD.h"
#include "MagneticField/GeomBuilder/src/volumeHandle.h"
#include "MagneticField/GeomBuilder/src/eLayer.h"

class MagESector;

class MagGeoBuilderFromDDD::eSector {
public:
  /// Constructor from list of volumes
  eSector(handles::const_iterator begin, handles::const_iterator end);

  /// Destructor
  ~eSector();

//   /// Return all volumes in this sector
//   const handles & getVolumes() const {return volumes;}

  /// Construct the MagESector upon request.
  MagESector* buildMagESector() const;

private:
  std::vector<eLayer> layers; // the layers in this sectors
  handles theVolumes;       // pointers to all volumes in the sector
  mutable MagESector* msector;
};
#endif
