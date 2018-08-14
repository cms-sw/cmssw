#ifndef Navigation_MuonNavigableLayer_H
#define Navigation_MuonNavigableLayer_H

/** \class MuonNavigableLayer
 *
 *  base class for MuonBarrelNavigableLayer and MuonForwardNavigable.
 *  trackingRange defines an MuonEtaRange for an FTS, 
 *  which is used for search compatible DetLayers.
 *
 *
 * \author : Chang Liu - Purdue University <Chang.Liu@cern.ch>
 *
 * Modification:
 *
 */

#include "RecoMuon/Navigation/interface/MuonDetLayerMap.h"
#include "RecoMuon/Navigation/interface/MuonEtaRange.h"

class DetLayer;
class BarrelDetLayer;

#include "TrackingTools/DetLayers/interface/NavigableLayer.h"


class MuonNavigableLayer : public NavigableLayer {

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

    MuonEtaRange trackingRange(const FreeTrajectoryState& fts) const;

    bool isInsideOut(const FreeTrajectoryState& fts) const;

};
#endif
