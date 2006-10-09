#ifndef CosmicMuonProducer_DirectMuonNavigation_H
#define CosmicMuonProducer_DirectMuonNavigation_H
/** \file DirectMuonNavigation
 *
 *  do a straight line extrapolation to
 *  find out compatible DetLayers with a given FTS 
 *
 *  $Date: 2006/06/28 15:41:12 $
 *  $Revision: 1.1 $
 *  \author Chang Liu  -  Purdue University
 */

#include "FWCore/Framework/interface/ESHandle.h"
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
    DirectMuonNavigation(const edm::ESHandle<MuonDetLayerGeometry>);

    DirectMuonNavigation* clone() const {
      return new DirectMuonNavigation(*this);
    }

    /* Destructor */ 
    ~DirectMuonNavigation() {}

    vector<const DetLayer*> 
      compatibleLayers( const FreeTrajectoryState& fts, 
                        PropagationDirection timeDirection) const;

  private:

    void inOutBarrel(const FreeTrajectoryState&, vector<const DetLayer*>&) const;
    void outInBarrel(const FreeTrajectoryState&, vector<const DetLayer*>&) const;

    void inOutForward(const FreeTrajectoryState&, vector<const DetLayer*>&) const;
    void outInForward(const FreeTrajectoryState&, vector<const DetLayer*>&) const; 

    void inOutBackward(const FreeTrajectoryState&, vector<const DetLayer*>&) const;
    void outInBackward(const FreeTrajectoryState&, vector<const DetLayer*>&) const;

    bool checkCompatible(const FreeTrajectoryState& fts,const BarrelDetLayer*) const;
    bool checkCompatible(const FreeTrajectoryState& fts,const ForwardDetLayer*) const;
    bool outward(const FreeTrajectoryState& fts) const;

    float epsilon_;
    edm::ESHandle<MuonDetLayerGeometry> theMuonDetLayerGeometry;

};
#endif
