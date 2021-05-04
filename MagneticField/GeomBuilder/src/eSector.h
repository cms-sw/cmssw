#ifndef eSector_H
#define eSector_H

/** \class MagGeoBuilderFromDDD::eSector
 *  A sector of volumes in the endcap.
 *  One sector is composed of several layers (eLayer)
 *
 *  \author N. Amapane - INFN Torino
 */

#include "eLayer.h"

class MagESector;

namespace magneticfield {
  class eSector {
  public:
    /// Constructor from list of volumes
    eSector(handles::const_iterator begin, handles::const_iterator end, bool debugFlag = false);

    /// Destructor
    ~eSector() = default;

    //   /// Return all volumes in this sector
    //   const handles & getVolumes() const {return volumes;}

    /// Construct the MagESector upon request.
    MagESector* buildMagESector() const;

  private:
    std::vector<eLayer> layers;  // the layers in this sectors
    handles theVolumes;          // pointers to all volumes in the sector
    mutable MagESector* msector;
    const bool debug;
  };
}  // namespace magneticfield

#endif
