#ifndef CondFormats_HGCalObjects_interface_TICLGeomLayersSoA_h
#define CondFormats_HGCalObjects_interface_TICLGeomLayersSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

// Per-layer replacement of RecHitTools::getPositionLayer, indexed by the
// absolute layer number (with offset). Row 0 is the barrel layer 0 entry.
//   z        endcap layer z at positive zside (0 beyond the endcap layers)
//   noseZ    HFNose layer z (0 when there is no nose geometry)
//   barrelX, barrelY  position of the first valid barrel cell of the layer
//                     (EB for layer 0, HB depths above), as in RecHitTools
GENERATE_SOA_LAYOUT(TICLGeomLayersSoALayout,
                    SOA_COLUMN(float, z),
                    SOA_COLUMN(float, noseZ),
                    SOA_COLUMN(float, barrelX),
                    SOA_COLUMN(float, barrelY))

using TICLGeomLayersSoA = TICLGeomLayersSoALayout<>;
using TICLGeomLayersSoAView = TICLGeomLayersSoA::View;
using TICLGeomLayersSoAConstView = TICLGeomLayersSoA::ConstView;

#endif  // CondFormats_HGCalObjects_interface_TICLGeomLayersSoA_h
