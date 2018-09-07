#ifndef Navigation_DirectMTDNavigation_H
#define Navigation_DirectMTDNavigation_H
/** \file DirectMTDNavigation
 *
 *  do a straight line extrapolation to
 *  find out compatible DetLayers with a given FTS 
 *
 *  \author Chang Liu  -  Purdue University
 */

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "RecoMTD/DetLayers/interface/MTDDetLayerGeometry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DirectMTDNavigation{

  public:

    /* Constructor */ 
    DirectMTDNavigation(const edm::ESHandle<MTDDetLayerGeometry>&);

    DirectMTDNavigation(const edm::ESHandle<MTDDetLayerGeometry>&, const edm::ParameterSet&);

    DirectMTDNavigation* clone() const {
      return new DirectMTDNavigation(*this);
    }

    /* Destructor */ 
    ~DirectMTDNavigation() {}

    std::vector<const DetLayer*> 
      compatibleLayers( const FreeTrajectoryState& fts, 
                        PropagationDirection timeDirection) const;


    std::vector<const DetLayer*>
      compatibleEndcapLayers( const FreeTrajectoryState& fts,
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

    edm::ESHandle<MTDDetLayerGeometry> theMTDDetLayerGeometry;
    float epsilon_;
    bool theEndcapFlag;
    bool theBarrelFlag;

};
#endif
