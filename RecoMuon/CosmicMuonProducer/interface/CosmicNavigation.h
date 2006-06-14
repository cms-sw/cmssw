#ifndef CosmicMuonProducer_CosmicNavigation_H
#define CosmicMuonProducer_CosmicNavigation_H
/** \file CosmicNavigation
 *
 *  $Date: $
 *  $Revision: $
 *  \author Chang Liu  -  Purdue University
 */

#include "Geometry/Surface/interface/BoundCylinder.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"

using namespace std;

class CosmicNavigation{

  public:

    /* Constructor */ 
    CosmicNavigation(const MuonDetLayerGeometry *);

    CosmicNavigation* clone() const {
      return new CosmicNavigation(*this);
    }

    /* Destructor */ 
    ~CosmicNavigation() {}

    vector<const DetLayer*> 
      compatibleLayers( const FreeTrajectoryState& fts, 
                        PropagationDirection timeDirection) const;

  private:
    void addBarrelLayer(BarrelDetLayer* dl);
    void addEndcapLayer(ForwardDetLayer* dl);

    bool checkCompatible(const FreeTrajectoryState& fts,const BarrelDetLayer*) const;
    bool checkCompatible(const FreeTrajectoryState& fts,const ForwardDetLayer*) const;
    bool outward(const FreeTrajectoryState& fts) const;

    float epsilon_;
    vector<const BarrelDetLayer*> theBarrelLayers; 
    vector<const ForwardDetLayer*> theForwardLayers; 
    vector<const ForwardDetLayer*> theBackwardLayers; 
    const MuonDetLayerGeometry * theMuonDetLayerGeometry; 
  protected:

};
#endif

