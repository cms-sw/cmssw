#ifndef RecoMuon_GlobalTrackingTools_StateSegmentMatcher_h
#define RecoMuon_GlobalTrackingTools_StateSegmentMatcher_h

/**
 *  Class: StateSegmentMatcher, Tsos4D, Tsos2DPhi, Tsos2DZed
 *
 *  Description:
 *  utility classes for the dynamical truncation algorithm
 *
 *  $Date: 2010/06/14 11:01:53 $
 *  $Revision: 1.1 $
 *
 *  Authors :
 *  D. Pagano & G. Bruno - UCL Louvain
 *
 **/

#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "RecoMuon/GlobalTrackingTools/interface/ChamberSegmentUtility.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"


class Tsos4D {
 public:

  Tsos4D(TrajectoryStateOnSurface* );

  // Returns the 4d vector
  AlgebraicVector4 paramVector() const;

  // Returns the 4x4 covariance matrix
  AlgebraicSymMatrix44 errorMatrix() const;

 private:
  AlgebraicVector4 tsos_4d;
  AlgebraicSymMatrix44 tsosErr_44;

};



class Tsos2DPhi {
 public:
  // Constructor of the class
  Tsos2DPhi(TrajectoryStateOnSurface* );

  // Returns the 2d vector
  AlgebraicVector2 paramVector() const;

  // Returns the 2x2 covariance matrix
  AlgebraicSymMatrix22 errorMatrix() const;

 private:
  AlgebraicVector2 tsos_2d_phi;
  AlgebraicSymMatrix22 tsosErr_22_phi;
};



class Tsos2DZed {
 public:

  Tsos2DZed(TrajectoryStateOnSurface* );

  // Returns the 2d vector
  AlgebraicVector2 paramVector() const;

  // Returns the 2x2 covariance matrix
  AlgebraicSymMatrix22 errorMatrix() const;

 private:
  AlgebraicVector2 tsos_2d_zed;
  AlgebraicSymMatrix22 tsosErr_22_zed;
};



class StateSegmentMatcher {

 public:
  
  // Perform the matching between a track state and a CSC segment
  StateSegmentMatcher(TrajectoryStateOnSurface*, CSCSegment*);

  // Perform the matching between a track state and a DT segment
  StateSegmentMatcher(TrajectoryStateOnSurface*, DTRecSegment4D*);

  // Returns the estimator value 
  double value();

 private:

  AlgebraicVector4 v1, v2;
  AlgebraicSymMatrix44 m1, m2;
  AlgebraicVector2 v1_2d, v2_2d;
  AlgebraicSymMatrix22 m1_2d, m2_2d;
  bool match2D;
  double estValue;
};


#endif


