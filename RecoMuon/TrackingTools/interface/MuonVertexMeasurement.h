#ifndef TrackingTools_MuonVertexMeasurement_H
#define TrackingTools_MuonVertexMeasurement_H

/**  \class MuonVertexMeasurement
  * 
  *   Class to store results of vertex extrapolation
  *
  *
  *   $Date:  $
  *   $Revision:  $
  *
  *   \author  N. Neumeister            Purdue University
  */


#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"

//              ---------------------
//              -- Class Interface --
//              ---------------------

class MuonVertexMeasurement {

  public:

    /// constructor
    MuonVertexMeasurement() : 
       trackerState(TrajectoryStateOnSurface()), 
       ipState(TrajectoryStateOnSurface()), 
       vertexState(TrajectoryStateOnSurface()),
       vertexMeasurement(TrajectoryMeasurement()), chi2(0.0) {}
    
    /// constructor
    MuonVertexMeasurement(const TrajectoryStateOnSurface& tk,
                          const TrajectoryStateOnSurface& ip,
                          const TrajectoryStateOnSurface& vx,
                          const TrajectoryMeasurement& tm,
                          const double c2) : 
       trackerState(tk), ipState(ip), vertexState(vx),
       vertexMeasurement(tm), chi2(c2) {}

    /// return trajectory at outer tracker surface
    TrajectoryStateOnSurface stateAtTracker() const { return trackerState; }

    /// return trajectory state at (transverse) impact point
    TrajectoryStateOnSurface stateAtIP() const { return ipState; }
    
    /// return trajectory state at vertex
    TrajectoryStateOnSurface stateAtVertex() const { return vertexState; }

    /// return Chi2 
    double chiSquared() const { return chi2; }

    /// return vertex measurement
    TrajectoryMeasurement measurement() const { return vertexMeasurement; }

  private:

    TrajectoryStateOnSurface trackerState;
    TrajectoryStateOnSurface ipState;
    TrajectoryStateOnSurface vertexState;
    TrajectoryMeasurement vertexMeasurement;

    double chi2;

};

#endif

