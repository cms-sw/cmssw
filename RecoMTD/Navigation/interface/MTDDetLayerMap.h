#ifndef MTDDetLayerMap_h
#define MTDDetLayerMap_h

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "RecoMTD/Navigation/interface/MTDEtaRange.h"

#include <map>
#include <functional>

/** \class MTDDetLayerComp
 *
 * Sort the of BarrelDetLayers by radius, ForwardDetLayer by |Z|.
 *
 * \author : L. Gray
 *
 * Adapted from MuonDetLayerComp
 *
 */

struct MTDDetLayerComp {
    bool operator()(const BarrelDetLayer* l1, const BarrelDetLayer* l2) const {
      if ( l1->specificSurface().radius() < l2->specificSurface().radius() ) return true;
      return false;
    }

    bool operator()(const ForwardDetLayer* l1, const ForwardDetLayer* l2) const {
      if ( fabs(l1->surface().position().z()) < fabs(l2->surface().position().z()) ) return true;
      return false;
    }
};


// FIXME: these names are too generic...
typedef std::map<const BarrelDetLayer*, MTDEtaRange, MTDDetLayerComp> MapB;
typedef std::map<const ForwardDetLayer*, MTDEtaRange, MTDDetLayerComp> MapE;
typedef MapB::const_iterator MapBI;
typedef MapE::const_iterator MapEI;

#endif
 
