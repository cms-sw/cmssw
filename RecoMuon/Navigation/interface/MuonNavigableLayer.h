#ifndef Navigation_MuonNavigableLayer_H
#define Navigation_MuonNavigableLayer_H

//   base class for MuonBarrelNavigableLayer and MuonForwardNavigable. 
//   TrackingRange defines an MuonEtaRange for an FTS, which is used for search compatible DetLayers.
//   $Date: $
//   $Revision: $
//   \auther Chang Liu             Purdue University

/* Collaborating Class Declarations */
#include "RecoMuon/Navigation/interface/MuonLayerSort.h"
#include "RecoMuon/Navigation/interface/MuonEtaRange.h"

class DetLayer;
class BarrelDetLayer;

/* Base Class Headers */
#include "TrackingTools/DetLayers/interface/NavigableLayer.h"

/* Class MuonNavigableLayer Interface */

class MuonNavigableLayer : public NavigableLayer {

  public:

    /// NavigableLayer interface
    virtual vector<const DetLayer*> nextLayers(PropagationDirection dir) const=0;

    /// NavigableLayer interface
    virtual vector<const DetLayer*> nextLayers(const FreeTrajectoryState& fts, 
                                               PropagationDirection dir) const=0;

    virtual vector<const DetLayer*> compatibleLayers(PropagationDirection dir) const=0;

    /// NavigableLayer interface
    virtual vector<const DetLayer*> compatibleLayers(const FreeTrajectoryState& fts,
                                               PropagationDirection dir) const=0;

    /// return DetLayer
    virtual DetLayer* detLayer() const=0;

    /// set DetLayer
    virtual void setDetLayer(DetLayer*)=0;

    MuonEtaRange TrackingRange(const FreeTrajectoryState& fts) const;

};
#endif
