#ifndef MuonDetLayerMap_h
#define MuonDetLayerMap_h

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "RecoMuon/Navigation/interface/MuonEtaRange.h"

#include <map>
#include <functional>

/** \class MuonDetLayerComp
 *
 * Sort the of BarrelDetLayers by radius, ForwardDetLayer by |Z|.
 *
 * $Date: 2006/10/13 15:06:01 $
 * $Revision: 1.1 $
 *
 * \author : Stefano Lacaprara, Nicola Amapane
 *
 *
 */

struct MuonDetLayerComp {
    bool operator()(BarrelDetLayer* l1, BarrelDetLayer* l2) const {
      if ( l1->specificSurface().radius() < l2->specificSurface().radius() ) return true;
      return false;
    }

    bool operator()(ForwardDetLayer* l1, ForwardDetLayer* l2) const {
      if ( fabs(l1->surface().position().z()) < fabs(l2->surface().position().z()) ) return true;
      return false;
    }
};


// FIXME: these names are too generic...
typedef std::map<BarrelDetLayer*, MuonEtaRange, MuonDetLayerComp> MapB;
typedef std::map<ForwardDetLayer*, MuonEtaRange, MuonDetLayerComp> MapE;
typedef MapB::const_iterator MapBI;
typedef MapE::const_iterator MapEI;

#endif
 
