#ifndef TrackingTools_MuonUpdatorAtVertex_H
#define TrackingTools_MuonUpdatorAtVertex_H

 /**  \class MuonUpdatorAtVertex
  *
  *   Extrapolate a muon trajectory to 
  *   a given vertex and 
  *   apply a vertex constraint
  *
  *   $Date: 2006/05/26 00:44:04 $
  *   $Revision: 1.1 $
  *
  *   \author   N. Neumeister            Purdue University
  *
  */



#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/CommonDetAlgo/interface/GlobalError.h"
#include "RecoMuon/TrackingTools/interface/MuonVertexMeasurement.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

class TrajectoryStateOnSurface;
class SteppingHelixPropagator;
class TransverseImpactPointExtrapolator;
class KFUpdator;
class MeasurementEstimator;
class Vertex;
class MagneticField;

class MuonUpdatorAtVertex {

  public:
 
    /// default constructor
    MuonUpdatorAtVertex(const MagneticField*);

    /// constructor
//    MuonUpdatorAtVertex(const Vertex&); 

    /// constructor
    MuonUpdatorAtVertex(const GlobalPoint, 
                        const GlobalError,const MagneticField*);
  
    /// destructor
    virtual ~MuonUpdatorAtVertex();

    /// return vertex measurement
//    MuonVertexMeasurement update(const RecTrack&) const;
    MuonVertexMeasurement update(const TrajectoryStateOnSurface&) const;
    
  private:
 
    GlobalPoint theVertexPos;
    GlobalError theVertexErr;

    SteppingHelixPropagator* thePropagator;
    TransverseImpactPointExtrapolator* theExtrapolator;
    KFUpdator* theUpdator;
    MeasurementEstimator* theEstimator;

};

#endif

