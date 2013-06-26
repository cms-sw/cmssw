#ifndef RecoMuon_GlobalTrackingTools_DirectTrackerNavigation_H
#define RecoMuon_GlobalTrackingTools_DirectTrackerNavigation_H

/** \file DirectTrackerNavigation
 *
 *  Find all DetLayers compatible with a given 
 *  Free Trajectory State by checking the eta range
 *
 *
 *  $Date: 2013/01/08 12:24:22 $
 *  $Revision: 1.3 $
 *
 *  \author Chang Liu  -  Purdue University
 */

#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

class DetLayer;
class BarrelDetLayer;
class ForwardDetLayer;
class FreeTrajectoryState;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DirectTrackerNavigation {

  public:

    /// constructor
    DirectTrackerNavigation(const edm::ESHandle<GeometricSearchTracker>&, 
                            bool outOnly = true);



    /// find compatible layers for a given trajectory state
    std::vector<const DetLayer*> 
      compatibleLayers(const FreeTrajectoryState& fts, 
                       PropagationDirection timeDirection) const;

  private:

    void inOutTOB(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const;

    void inOutTIB(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const;

    void inOutPx(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const;

    void inOutFTEC(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const;

    void inOutFTID(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const;

    void inOutFPx(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const;

    void inOutBTEC(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const;

    void inOutBTID(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const;

    void inOutBPx(const FreeTrajectoryState&, std::vector<const DetLayer*>&) const;

    bool checkCompatible(const FreeTrajectoryState&, const BarrelDetLayer*) const;

    bool checkCompatible(const FreeTrajectoryState&, const ForwardDetLayer*) const;

    bool outward(const FreeTrajectoryState&) const;

    float calculateEta(float r, float z) const;

  private:

    edm::ESHandle<GeometricSearchTracker> theGeometricSearchTracker;

    bool theOutLayerOnlyFlag;

    float theEpsilon;

};
#endif
