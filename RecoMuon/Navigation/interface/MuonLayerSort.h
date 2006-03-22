#ifndef Navigation_MuonLayerSort_H
#define Navigation_MuonLayerSort_H

//   Ported from ORCA.

//   $Date: $
//   $Revision: $

/* Collaborating Class Declarations */
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "RecoMuon/Navigation/interface/MuonEtaRange.h"
#include <map>
#include <functional>

/// Sort the list of layers in the barrel region
class LayerBarrel_comp : binary_function<const BarrelDetLayer*, const BarrelDetLayer*, bool> {
  public:
    bool operator()( const  BarrelDetLayer* l1, const  BarrelDetLayer* l2) const {
      if ( l1->specificSurface().radius() < l2->specificSurface().radius() ) return true;
      return false;
    }

};

/// Sort the list of layers in the forward regions
class LayerEndcap_comp : binary_function<const ForwardDetLayer*, const ForwardDetLayer*, bool> {
  public:
    bool operator()( const  ForwardDetLayer* l1, const  ForwardDetLayer* l2) const {
      if ( fabs(l1->surface().position().z()) < fabs(l2->surface().position().z()) ) return true;
      return false;
    }

};
typedef map<const BarrelDetLayer*, MuonEtaRange, LayerBarrel_comp> MapB;
typedef map<const ForwardDetLayer*, MuonEtaRange, LayerEndcap_comp> MapE;
typedef MapB::const_iterator MapBI;
typedef MapE::const_iterator MapEI;

#endif
