#ifndef TrackingTools_MuonUpdatorAtVertex_H
#define TrackingTools_MuonUpdatorAtVertex_H

 /**  \class MuonUpdatorAtVertex
  *
  *   Extrapolate a muon trajectory to 
  *   a given vertex and 
  *   apply a vertex constraint
  *
  *   $Date: 2006/08/31 18:24:17 $
  *   $Revision: 1.7 $
  *
  *   \author   N. Neumeister            Purdue University
  *
  */

#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/CommonDetAlgo/interface/GlobalError.h"
#include "RecoMuon/TrackingTools/interface/MuonVertexMeasurement.h"
#include "FWCore/Framework/interface/EventSetup.h"

class TrajectoryStateOnSurface;
class Propagator;
class TransverseImpactPointExtrapolator;
class KFUpdator;
class MeasurementEstimator;
class MuonServiceProxy;

namespace edm {class ParameterSet; class EventSetup;}

class MuonUpdatorAtVertex {

 public:
 
  /// constructor from parameter set and MuonServiceProxy
  MuonUpdatorAtVertex(const edm::ParameterSet&,const MuonServiceProxy *);
  
  /// default constructor
  MuonUpdatorAtVertex();

  /// constructor from propagator
  MuonUpdatorAtVertex(const Propagator&);

  /// destructor
  virtual ~MuonUpdatorAtVertex();

  /// get Propagator for outside tracker, SteppingHelixPropagator as default
  /// anyDirection
  std::auto_ptr<Propagator> propagator() const;
    
  /// return vertex measurement
  MuonVertexMeasurement update(const TrajectoryStateOnSurface&) const;

  /// only return the state on outer tracker surface
  TrajectoryStateOnSurface stateAtTracker(const TrajectoryStateOnSurface&) const;

  void setVertex(const GlobalPoint&, const GlobalError&);

    
 private:

  const MuonServiceProxy *theService;
 
  GlobalPoint theVertexPos;
  GlobalError theVertexErr;

  TransverseImpactPointExtrapolator* theExtrapolator;
  KFUpdator* theUpdator;
  MeasurementEstimator* theEstimator;
  std::string theOutPropagatorName;
  std::string theInPropagatorName;
};

#endif
