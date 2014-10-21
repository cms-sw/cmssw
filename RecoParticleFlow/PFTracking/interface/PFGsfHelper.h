#ifndef PFGsfHelper_H
#define PFGsfHelper_H

#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianStateTransform.h"


/// \brief Abstract
/*!
\author Daniele Benedetti
\date January 2008

 PFGsfHelper is a class used to give back
 the p mode and the DP and sigmaDP for each trajectory on surface. 
 
 Other utilities: 
*/

class TrajectoryMeasurement; 
class PFGsfHelper{
  
 public:
  PFGsfHelper ( const TrajectoryMeasurement&);
  ~PFGsfHelper();

  GlobalVector computeP(bool ComputeMode) const;
  bool isValid () const;
  double fittedDP () const;
  double sigmafittedDP() const;

 private:
  
  void computeQpMode (const TrajectoryStateOnSurface tsos,
		      AlgebraicVector5& parameters, AlgebraicSymMatrix55& covariance) const;
  

  float mode_Px;
  float mode_Py;
  float mode_Pz;
  bool Valid;
  double dp;
  double sigmaDp;
  TrajectoryStateOnSurface theUpdateState;
  TrajectoryStateOnSurface theForwardState;
  TrajectoryStateOnSurface theBackwardState;
};

#endif
