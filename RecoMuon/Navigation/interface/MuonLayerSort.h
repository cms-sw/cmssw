#ifndef Navigation_MuonLayerSort_H
#define Navigation_MuonLayerSort_H

/** \class MuonLayerSort
 *
 *  class to create a map of DetLayers
 *
 * $Date: 2006/06/04 18:27:00 $
 * $Revision: 1.3 $
 *
 * \author : Stefano Lacaprara - INFN Padova <stefano.lacaprara@pd.infn.it>
 *
 * Modification:
 *
 */

/* Collaborating Class Declarations */
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "RecoMuon/Navigation/interface/MuonEtaRange.h"
#include <map>
#include <functional>

/// Sort the list of layers in the barrel region
class LayerBarrel_comp : std::binary_function<BarrelDetLayer*, BarrelDetLayer*, bool> {
  public:
    bool operator()(BarrelDetLayer* l1, BarrelDetLayer* l2) const {
      if ( l1->specificSurface().radius() < l2->specificSurface().radius() ) return true;
      return false;
    }

};

/// Sort the list of layers in the forward regions
class LayerEndcap_comp : std::binary_function<ForwardDetLayer*, ForwardDetLayer*, bool> {
  public:
    bool operator()(ForwardDetLayer* l1, ForwardDetLayer* l2) const {
      if ( fabs(l1->surface().position().z()) < fabs(l2->surface().position().z()) ) return true;
      return false;
    }

};
typedef std::map<BarrelDetLayer*, MuonEtaRange, LayerBarrel_comp> MapB;
typedef std::map<ForwardDetLayer*, MuonEtaRange, LayerEndcap_comp> MapE;
typedef MapB::const_iterator MapBI;
typedef MapE::const_iterator MapEI;

#endif
