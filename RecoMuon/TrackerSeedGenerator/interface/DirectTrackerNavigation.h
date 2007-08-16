#ifndef RecoMuon_DirectTrackerNavigation_H
#define RecoMuon_DirectTrackerNavigation_H

/** \file DirectTrackerNavigation
 *
 *  find out compatible DetLayers with a given FTS 
 *  by checking eta region
 *
 *  $Date: 2007/05/11 15:02:11 $
 *  $Revision: 1.1 $
 *  \author Chang Liu  -  Purdue University
 */

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

class DirectTrackerNavigation{

  public:

    /* Constructor */ 
    DirectTrackerNavigation(const edm::ESHandle<GeometricSearchTracker>&, bool outOnly = true);

    DirectTrackerNavigation* clone() const {
      return new DirectTrackerNavigation(*this);
    }

    /* Destructor */ 
    ~DirectTrackerNavigation() {}

    std::vector<const DetLayer*> 
      compatibleLayers( const FreeTrajectoryState& fts, 
                        PropagationDirection timeDirection) const;

  private:

    void inOutTOB(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const;
//    void outInTOB(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const;

    void inOutTIB(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const;
//    void outInTIB(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const;

    void inOutPx(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const;
//    void outInPx(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const;


    void inOutFTEC(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const;
//    void outInFTEC(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const; 
    void inOutFTID(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const;
//    void outInFTID(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const;
    void inOutFPx(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const;
//    void outInFPx(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const;

    void inOutBTEC(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const;
//    void outInBTEC(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const;
    void inOutBTID(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const;
//    void outInBTID(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const;
    void inOutBPx(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const;
//    void outInBPx(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const;

    bool checkCompatible(const FreeTrajectoryState& fts,const BarrelDetLayer*) const;
    bool checkCompatible(const FreeTrajectoryState& fts,const ForwardDetLayer*) const;
    bool outward(const FreeTrajectoryState& fts) const;

    float calculateEta(float r, float z) const;

    float epsilon_;

    edm::ESHandle<GeometricSearchTracker> theGeometricSearchTracker;

    bool theOutLayerOnlyFlag;

};
#endif
