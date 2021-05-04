#ifndef eLayer_H
#define eLayer_H

/** \class MagGeoBuilderFromDDD::eLayer
 *  A layer of volumes in an endcap sector.
 *
 *  \author N. Amapane - INFN Torino
 */

#include "bSector.h"

class MagELayer;

namespace magneticfield {
  class eLayer {
  public:
    /// Constructor from list of volumes
    eLayer(handles::const_iterator begin, handles::const_iterator end);

    /// Destructor
    ~eLayer() = default;

    //   /// Return the list of all volumes.
    //   const handles & volumes() const {return theVolumes;}

    /// Construct the MagELayer upon request.
    MagELayer* buildMagELayer() const;

  private:
    handles theVolumes;  // pointer to all volumes in this layer
    mutable MagELayer* mlayer;
  };
}  // namespace magneticfield

#endif
