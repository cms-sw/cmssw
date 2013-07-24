#ifndef Navigation_MuonNavigableLayer_H
#define Navigation_MuonNavigableLayer_H

/** \class MuonNavigableLayer
 *
 *  base class for MuonBarrelNavigableLayer and MuonForwardNavigable.
 *  trackingRange defines an MuonEtaRange for an FTS, 
 *  which is used for search compatible DetLayers.
 *
 * $Date: 2007/01/29 16:24:17 $
 * $Revision: 1.8 $
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
    virtual std::vector<const DetLayer*> nextLayers(NavigationDirection dir) const=0;

    /// NavigableLayer interface
    virtual std::vector<const DetLayer*> nextLayers(const FreeTrajectoryState& fts, 
                                               PropagationDirection dir) const=0;

    virtual std::vector<const DetLayer*> compatibleLayers(NavigationDirection dir) const=0;

    /// NavigableLayer interface
    virtual std::vector<const DetLayer*> compatibleLayers(const FreeTrajectoryState& fts,
                                               PropagationDirection dir) const=0;

    /// return DetLayer
    virtual DetLayer* detLayer() const=0;

    /// set DetLayer
    virtual void setDetLayer(DetLayer*)=0;

    MuonEtaRange trackingRange(const FreeTrajectoryState& fts) const;

    bool isInsideOut(const FreeTrajectoryState& fts) const;

};
#endif
