#ifndef CosmicMuonProducer_DirectMuonNavigation_H
#define CosmicMuonProducer_DirectMuonNavigation_H
/** \file DirectMuonNavigation
 *
 *  do a straight line extrapolation to
 *  find out compatible DetLayers with a given FTS 
 *
 *  $Date: 2006/09/01 19:02:21 $
 *  $Revision: 1.2 $
 *  \author Chang Liu  -  Purdue University
 */

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Surface/interface/BoundCylinder.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"

class DirectMuonNavigation{

  public:

    /* Constructor */ 
    DirectMuonNavigation(const edm::ESHandle<MuonDetLayerGeometry>);

    DirectMuonNavigation* clone() const {
      return new DirectMuonNavigation(*this);
    }

    /* Destructor */ 
    ~DirectMuonNavigation() {}

    std::vector<const DetLayer*> 
      compatibleLayers( const FreeTrajectoryState& fts, 
                        PropagationDirection timeDirection) const;

  private:

    void inOutBarrel(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const;
    void outInBarrel(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const;

    void inOutForward(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const;
    void outInForward(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const; 

    void inOutBackward(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const;
    void outInBackward(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const;

    bool checkCompatible(const FreeTrajectoryState& fts,const BarrelDetLayer*) const;
    bool checkCompatible(const FreeTrajectoryState& fts,const ForwardDetLayer*) const;
    bool outward(const FreeTrajectoryState& fts) const;

    float epsilon_;
    edm::ESHandle<MuonDetLayerGeometry> theMuonDetLayerGeometry;

};
#endif
