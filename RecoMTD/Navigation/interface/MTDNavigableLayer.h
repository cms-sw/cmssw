#ifndef Navigation_MTDNavigableLayer_H
#define Navigation_MTDNavigableLayer_H

/** \class MTDNavigableLayer
 *
 *  base class for BTLNavigableLayer and ETLNavigableLayer.
 *  trackingRange defines an MTDEtaRange for an FTS, 
 *  which is used for search compatible DetLayers.
 *
 *
 * \author : L. Gray - FNAL
 *
 * Modification:
 *
 */

#include "RecoMTD/Navigation/interface/MTDDetLayerMap.h"
#include "RecoMTD/Navigation/interface/MTDEtaRange.h"

class DetLayer;
class BarrelDetLayer;

#include "TrackingTools/DetLayers/interface/NavigableLayer.h"


class MTDNavigableLayer : public NavigableLayer {

  public:

    /// NavigableLayer interface
    std::vector<const DetLayer*> nextLayers(NavigationDirection dir) const override =0;

    /// NavigableLayer interface
    std::vector<const DetLayer*> nextLayers(const FreeTrajectoryState& fts, 
                                               PropagationDirection dir) const override =0;

    std::vector<const DetLayer*> compatibleLayers(NavigationDirection dir) const override =0;

    /// NavigableLayer interface
    std::vector<const DetLayer*> compatibleLayers(const FreeTrajectoryState& fts,
                                               PropagationDirection dir) const override =0;

    /// return DetLayer
    const DetLayer* detLayer() const override =0;

    /// set DetLayer
    void setDetLayer(const DetLayer*) override =0;

    MTDEtaRange trackingRange(const FreeTrajectoryState& fts) const;

    bool isInsideOut(const FreeTrajectoryState& fts) const;

};
#endif
