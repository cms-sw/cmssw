#ifndef Navigation_ETLNavigableLayer_H
#define Navigation_ETLNavigableLayer_H

/** \class ETLNavigableLayer
 *
 *  Navigable layer for Forward Muon
 *
 *
 * \author : Stefano Lacaprara - INFN Padova <stefano.lacaprara@pd.infn.it>
 *
 * Modification:
 * Chang Liu:
 *  compatibleLayers(dir) and compatibleLayers(fts, dir) are added,
 *  which return ALL DetLayers that are compatible with a given DetLayer.
 */

/* Collaborating Class Declarations */
#include "RecoMTD/Navigation/interface/MTDDetLayerMap.h"
#include "RecoMTD/Navigation/interface/MTDEtaRange.h"

class DetLayer;
class ForwardDetLayer;

/* Base Class Headers */
#include "RecoMTD/Navigation/interface/MTDNavigableLayer.h"

/* C++ Headers */

/* ====================================================================== */

/* Class ETLNavigableLayer Interface */

class ETLNavigableLayer : public MTDNavigableLayer {

  public:

    ETLNavigableLayer(const ForwardDetLayer* fdl,
		      const MapB& innerBarrel, 
		      const MapE& outerEndcap,
		      const MapE& innerEndcap,
		      const MapB& allInnerBarrel,
		      const MapE& allOuterEndcap,
		      const MapE& allInnerEndcap) :
      theDetLayer(fdl),
      theInnerBarrelLayers(innerBarrel),
      theOuterEndcapLayers(outerEndcap),
      theInnerEndcapLayers(innerEndcap),
      theAllInnerBarrelLayers(allInnerBarrel), 
      theAllOuterEndcapLayers(allOuterEndcap),
      theAllInnerEndcapLayers(allInnerEndcap)  {}

    /// Constructor with outer layers only
    ETLNavigableLayer(const ForwardDetLayer* fdl,
		      const MapE& outerEndcap) :
      theDetLayer(fdl),
      theOuterEndcapLayers(outerEndcap) {}
    /// Constructor with all outer layers only
    ETLNavigableLayer(const ForwardDetLayer* fdl,
		      const MapE& outerEndcap, 
		      const MapE& allOuterEndcap) :
      theDetLayer(fdl),
      theOuterEndcapLayers(outerEndcap),
      theAllOuterEndcapLayers(allOuterEndcap) {}


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

    /// Operations
    MapE getOuterEndcapLayers() const { return theOuterEndcapLayers; }
    MapE getInnerEndcapLayers() const { return theInnerEndcapLayers; }
    MapB getInnerBarrelLayers() const { return theInnerBarrelLayers; }

    MapE getAllOuterEndcapLayers() const { return theAllOuterEndcapLayers; }
    MapE getAllInnerEndcapLayers() const { return theAllInnerEndcapLayers; }
    MapB getAllInnerBarrelLayers() const { return theAllInnerBarrelLayers; }

    /// set inward links
    void setInwardLinks(const MapB&, const MapE&);
    void setInwardCompatibleLinks(const MapB&, const MapE&);

  private:

    void pushResult(std::vector<const DetLayer*>& result, 
                    const MapB& map) const;

    void pushResult(std::vector<const DetLayer*>& result,
                     const MapE& map) const;

    void pushResult(std::vector<const DetLayer*>& result, 
                    const MapB& map, 
                    const FreeTrajectoryState& fts) const;

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

    const ForwardDetLayer* theDetLayer;
    MapB theInnerBarrelLayers;
    MapE theOuterEndcapLayers;
    MapE theInnerEndcapLayers;
    MapB theAllInnerBarrelLayers;
    MapE theAllOuterEndcapLayers;
    MapE theAllInnerEndcapLayers;

};
#endif
