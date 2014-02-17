#ifndef bRod_H
#define bRod_H

/** \class MagGeoBuilderFromDDD::bRod
 *  A rod of volumes in a barrel sector.
 *  A rod is made of several "slabs".
 *
 *  $Date: 2005/09/27 15:15:52 $
 *  $Revision: 1.2 $
 *  \author N. Amapane - INFN Torino
 */

#include "MagneticField/GeomBuilder/src/MagGeoBuilderFromDDD.h"
#include "MagneticField/GeomBuilder/src/volumeHandle.h"
#include "MagneticField/GeomBuilder/src/bSlab.h"

class MagBRod;

class MagGeoBuilderFromDDD::bRod {
public:
  /// Constructor from list of volumes
  bRod(handles::const_iterator begin, handles::const_iterator end);

  /// Destructor
  ~bRod();
 
  /// Distance from center along sector normal.
  const float RN() const {
    return volumes.front()->RN();
  }

  /// Construct the MagBRod upon request.
  MagBRod* buildMagBRod() const;

private:
  std::vector<bSlab> slabs;
  handles volumes; // pointers to all volumes in the rod
  mutable MagBRod* mrod;
};

#endif
