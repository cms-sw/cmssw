#ifndef CosmicMuonProducer_DirectMuonNavigation_H
#define CosmicMuonProducer_DirectMuonNavigation_H
/** \file DirectMuonNavigation
 *
 *  $Date: 2006/06/14 00:04:49 $
 *  $Revision: 1.1 $
 *  \author Chang Liu  -  Purdue University
 */

#include "Geometry/Surface/interface/BoundCylinder.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"

using namespace std;

class DirectMuonNavigation{

  public:

    /* Constructor */ 
    DirectMuonNavigation(const MuonDetLayerGeometry *);

    DirectMuonNavigation* clone() const {
      return new DirectMuonNavigation(*this);
    }

    /* Destructor */ 
    ~DirectMuonNavigation() {}

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

