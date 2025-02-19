#ifndef Navigation_MuonForwardNavigableLayer_H
#define Navigation_MuonForwardNavigableLayer_H

/** \class MuonForwardNavigableLayer
 *
 *  Navigable layer for Forward Muon
 *
 * $Date: 2007/01/18 13:28:36 $
 * $Revision: 1.7 $
 *
 * \author : Stefano Lacaprara - INFN Padova <stefano.lacaprara@pd.infn.it>
 *
 * Modification:
 * Chang Liu:
 *  compatibleLayers(dir) and compatibleLayers(fts, dir) are added,
 *  which return ALL DetLayers that are compatible with a given DetLayer.
 */

/* Collaborating Class Declarations */
#include "RecoMuon/Navigation/interface/MuonDetLayerMap.h"
#include "RecoMuon/Navigation/interface/MuonEtaRange.h"

class DetLayer;
class ForwardDetLayer;

/* Base Class Headers */
#include "RecoMuon/Navigation/interface/MuonNavigableLayer.h"

/* C++ Headers */

/* ====================================================================== */

/* Class MuonForwardNavigableLayer Interface */

class MuonForwardNavigableLayer : public MuonNavigableLayer {

  public:

    MuonForwardNavigableLayer(ForwardDetLayer* fdl,
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
    MuonForwardNavigableLayer(ForwardDetLayer* fdl,
                              const MapE& outerEndcap) :
      theDetLayer(fdl),
      theOuterEndcapLayers(outerEndcap) {}
    /// Constructor with all outer layers only
    MuonForwardNavigableLayer(ForwardDetLayer* fdl,
                              const MapE& outerEndcap, 
                              const MapE& allOuterEndcap) :
      theDetLayer(fdl),
      theOuterEndcapLayers(outerEndcap),
      theAllOuterEndcapLayers(allOuterEndcap) {}


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

    ForwardDetLayer* theDetLayer;
    MapB theInnerBarrelLayers;
    MapE theOuterEndcapLayers;
    MapE theInnerEndcapLayers;
    MapB theAllInnerBarrelLayers;
    MapE theAllOuterEndcapLayers;
    MapE theAllInnerEndcapLayers;

};
#endif
