#ifndef Navigation_MuonBarrelNavigableLayer_H
#define Navigation_MuonBarrelNavigableLayer_H

/** \class MuonBarrelNavigableLayer
 *
 *  Navigable layer for Barrel Muon 
 *
 *  $Date: 2007/01/18 13:28:36 $
 *  $Revision: 1.6 $
 *
 * \author : Stefano Lacaprara - INFN Padova <stefano.lacaprara@pd.infn.it>
 *
 * Modification:
 *
 * Chang Liu:
 * compatibleLayers(dir) and compatibleLayers(fts, dir) are added,
 * which returns ALL DetLayers that are compatible with a given DetLayer.
 *  
 */


/* Collaborating Class Declarations */
#include "RecoMuon/Navigation/interface/MuonDetLayerMap.h"
#include "RecoMuon/Navigation/interface/MuonEtaRange.h"

class DetLayer;
class BarrelDetLayer;

/* Base Class Headers */
#include "RecoMuon/Navigation/interface/MuonNavigableLayer.h"
/* C++ Headers */

/* ====================================================================== */

/* Class MuonBarrelNavigableLayer Interface */

class MuonBarrelNavigableLayer : public MuonNavigableLayer {

  public:

    /// Constructor 
    MuonBarrelNavigableLayer(BarrelDetLayer* bdl, 
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

    MuonBarrelNavigableLayer(BarrelDetLayer* bdl,
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
    MuonBarrelNavigableLayer(BarrelDetLayer* bdl, 
                             const MapB& outerBarrel,
                             const MapE& outerBackward,
                             const MapE& outerForward) :
      theDetLayer(bdl), 
      theOuterBarrelLayers(outerBarrel),
      theOuterBackwardLayers(outerBackward),
      theOuterForwardLayers(outerForward) { }

    MuonBarrelNavigableLayer(BarrelDetLayer* bdl,
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
    virtual std::vector<const DetLayer*> nextLayers(NavigationDirection dir) const;

    /// NavigableLayer interface
    virtual std::vector<const DetLayer*> nextLayers(const FreeTrajectoryState& fts, 
                                               PropagationDirection dir) const;

    virtual std::vector<const DetLayer*> compatibleLayers(NavigationDirection dir) const;

    /// NavigableLayer interface
    virtual std::vector<const DetLayer*> compatibleLayers(const FreeTrajectoryState& fts,
                                               PropagationDirection dir) const;

    /// return DetLayer
    virtual DetLayer* detLayer() const;

    /// set DetLayer
    virtual void setDetLayer(DetLayer*);

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

    BarrelDetLayer* theDetLayer;
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
