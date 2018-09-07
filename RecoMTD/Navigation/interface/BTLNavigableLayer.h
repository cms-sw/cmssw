#ifndef Navigation_BTLNavigableLayer_H
#define Navigation_BTLNavigableLayer_H

/** \class BTLNavigableLayer
 *
 *  Navigable layer for Barrel Timing Layer. 
 *  Taken from MuonBarrelNavigableLayer.
 *
 *
 * \author : L. Gray - FNAL
 * 
 */


/* Collaborating Class Declarations */
#include "RecoMTD/Navigation/interface/MTDDetLayerMap.h"
#include "RecoMTD/Navigation/interface/MTDEtaRange.h"

class DetLayer;
class BarrelDetLayer;

/* Base Class Headers */
#include "RecoMTD/Navigation/interface/MTDNavigableLayer.h"
/* C++ Headers */

/* ====================================================================== */

/* Class BTLNavigableLayer Interface */

class BTLNavigableLayer : public MTDNavigableLayer {

  public:

    /// Constructor 
    BTLNavigableLayer(BarrelDetLayer* bdl, 
		      const MapB& outerBarrel, 
		      const MapB& innerBarrel, 
		      const MapE& outerBackward,
		      const MapE& outerForward,
		      const MapE& innerBackward,
		      const MapE& innerForward) :
  theDetLayer(bdl), 
  theOuterBarrelLayers(outerBarrel),
  theInnerBarrelLayers(innerBarrel), 
  theOuterBackwardLayers(outerBackward),
  theInnerBackwardLayers(innerBackward),
  theOuterForwardLayers(outerForward),
  theInnerForwardLayers(innerForward) {}

    BTLNavigableLayer(BarrelDetLayer* bdl,
		      const MapB& outerBarrel,
		      const MapB& innerBarrel,
		      const MapE& outerBackward,
		      const MapE& outerForward,
		      const MapE& innerBackward,
		      const MapE& innerForward,
		      const MapB& allOuterBarrel,
		      const MapB& allInnerBarrel,
		      const MapE& allOuterBackward,
		      const MapE& allOuterForward,
		      const MapE& allInnerBackward,
		      const MapE& allInnerForward) :
      theDetLayer(bdl),
      theOuterBarrelLayers(outerBarrel),
      theInnerBarrelLayers(innerBarrel),
      theOuterBackwardLayers(outerBackward),
      theInnerBackwardLayers(innerBackward),
      theOuterForwardLayers(outerForward),
      theInnerForwardLayers(innerForward), 
      theAllOuterBarrelLayers(allOuterBarrel),
      theAllInnerBarrelLayers(allInnerBarrel),
      theAllOuterBackwardLayers(allOuterBackward),
      theAllInnerBackwardLayers(allInnerBackward),
      theAllOuterForwardLayers(allOuterForward),
      theAllInnerForwardLayers(allInnerForward) {}

    /// Constructor with outer layers only
    BTLNavigableLayer(BarrelDetLayer* bdl, 
                             const MapB& outerBarrel,
                             const MapE& outerBackward,
                             const MapE& outerForward) :
      theDetLayer(bdl), 
      theOuterBarrelLayers(outerBarrel),
      theOuterBackwardLayers(outerBackward),
      theOuterForwardLayers(outerForward) { }

    BTLNavigableLayer(const BarrelDetLayer* bdl,
                             const MapB& outerBarrel,
                             const MapE& outerBackward,
                             const MapE& outerForward,
                             const MapB& allOuterBarrel,
                             const MapE& allOuterBackward,
                             const MapE& allOuterForward) :
      theDetLayer(bdl),
      theOuterBarrelLayers(outerBarrel),
      theOuterBackwardLayers(outerBackward),
      theOuterForwardLayers(outerForward),
      theAllOuterBarrelLayers(allOuterBarrel),
      theAllOuterBackwardLayers(allOuterBackward),
      theAllOuterForwardLayers(allOuterForward) {}

    /// NavigableLayer interface
    std::vector<const DetLayer*> nextLayers(NavigationDirection dir) const override;

    /// NavigableLayer interface
    std::vector<const DetLayer*> nextLayers(const FreeTrajectoryState& fts, 
                                               PropagationDirection dir) const override;

    std::vector<const DetLayer*> compatibleLayers(NavigationDirection dir) const override;

    /// NavigableLayer interface
    std::vector<const DetLayer*> compatibleLayers(const FreeTrajectoryState& fts,
                                               PropagationDirection dir) const override;

    /// return DetLayer
    const DetLayer* detLayer() const override;

    /// set DetLayer
    void setDetLayer(const DetLayer*) override;

    MapB getOuterBarrelLayers() const { return theOuterBarrelLayers; }
    MapB getInnerBarrelLayers() const { return theInnerBarrelLayers; }
    MapE getOuterBackwardLayers() const { return theOuterBackwardLayers; }
    MapE getInnerBackwardLayers() const { return theInnerBackwardLayers; }
    MapE getOuterForwardLayers() const { return theOuterForwardLayers; }
    MapE getInnerForwardLayers() const { return theInnerForwardLayers; }

    MapB getAllOuterBarrelLayers() const { return theAllOuterBarrelLayers; }
    MapB getAllInnerBarrelLayers() const { return theAllInnerBarrelLayers; }
    MapE getAllOuterBackwardLayers() const { return theAllOuterBackwardLayers; }
    MapE getAllInnerBackwardLayers() const { return theAllInnerBackwardLayers; }
    MapE getAllOuterForwardLayers() const { return theAllOuterForwardLayers; }
    MapE getAllInnerForwardLayers() const { return theAllInnerForwardLayers; }

    /// set inward links
    void setInwardLinks(const MapB&);
    void setInwardCompatibleLinks(const MapB&);

  private:

    void pushResult(std::vector<const DetLayer*>& result,
                    const MapB& map) const;

    void pushResult(std::vector<const DetLayer*>& result,
                    const MapE& map) const;

    void pushResult(std::vector<const DetLayer*>& result,
                    const MapB& map, const
                    FreeTrajectoryState& fts) const;

    void pushResult(std::vector<const DetLayer*>& result,
                    const MapE& map, const
                    FreeTrajectoryState& fts) const;
    void pushCompatibleResult(std::vector<const DetLayer*>& result,
                    const MapB& map, const
                    FreeTrajectoryState& fts) const;

    void pushCompatibleResult(std::vector<const DetLayer*>& result,
                    const MapE& map, const
                    FreeTrajectoryState& fts) const;

  private:

    const BarrelDetLayer* theDetLayer;
    MapB theOuterBarrelLayers;
    MapB theInnerBarrelLayers;
    MapE theOuterBackwardLayers;
    MapE theInnerBackwardLayers;
    MapE theOuterForwardLayers;
    MapE theInnerForwardLayers;
    MapB theAllOuterBarrelLayers;
    MapB theAllInnerBarrelLayers;
    MapE theAllOuterBackwardLayers;
    MapE theAllInnerBackwardLayers;
    MapE theAllOuterForwardLayers;
    MapE theAllInnerForwardLayers;

};
#endif
