#ifndef DTLAYERTYPE_H
#define DTLAYERTYPE_H

/** \class DTLayerType
 *
 *  DetType for a Drift Tube GeomDetUnit (the DTLayer).
 *
 *  \author : Stefano Lacaprara - INFN Padova <stefano.lacaprara@pd.infn.it>
 *
 */

/* Base Class Headers */
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

class DTLayerType : public GeomDetType {
public:
  /* Constructor */
  DTLayerType();

  /* Operations */
  const Topology& topology() const override;
};
#endif  // DTLAYERTYPE_H
