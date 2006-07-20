#ifndef TrackingTools_MuonUpdatorAtVertex_H
#define TrackingTools_MuonUpdatorAtVertex_H

 /**  \class MuonUpdatorAtVertex
  *
  *   Extrapolate a muon trajectory to 
  *   a given vertex and 
  *   apply a vertex constraint
  *
  *   $Date: 2006/06/10 19:53:14 $
  *   $Revision: 1.2 $
  *
  *   \author   N. Neumeister            Purdue University
  *
  */



#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/CommonDetAlgo/interface/GlobalError.h"
#include "RecoMuon/TrackingTools/interface/MuonVertexMeasurement.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

class TrajectoryStateOnSurface;
class TransverseImpactPointExtrapolator;
class KFUpdator;
class MeasurementEstimator;

namespace edm {class ParameterSet; class EventSetup;}

class MuonUpdatorAtVertex {

  public:
 
    /// constructor from parameter set
    MuonUpdatorAtVertex(const edm::ParameterSet&);

    /// default constructor
    MuonUpdatorAtVertex();

    /// constructor
//    MuonUpdatorAtVertex(const Vertex&); 

    /// destructor
    virtual ~MuonUpdatorAtVertex();

    void init(const edm::EventSetup&);

    /// return vertex measurement
    MuonVertexMeasurement update(const TrajectoryStateOnSurface&) const;

    void setVertex(const GlobalPoint, const GlobalError);

    
  private:
 
    GlobalPoint theVertexPos;
    GlobalError theVertexErr;

    Propagator* thePropagator;
    TransverseImpactPointExtrapolator* theExtrapolator;
    KFUpdator* theUpdator;
    MeasurementEstimator* theEstimator;
    std::string thePropagatorName;

};

#endif

