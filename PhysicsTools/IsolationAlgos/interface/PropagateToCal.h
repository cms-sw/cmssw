#ifndef IsolationUtils_PropagateToCal_h
#define IsolationUtils_PropagateToCal_h
/* \class PropagateToCal
 *
 * \author Christian Autermann, U Hamburg
 *
 * class extrapolats a charged particle to the calorimeter surface 
 * using the SteppingHelixPropagator.
 *
 */
#include <algorithm>
#include <vector>
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"

class MagneticField;

class PropagateToCal {
public:
  PropagateToCal();
  ~PropagateToCal();
  PropagateToCal(double radius, double minZ, double maxZ, bool theIgnoreMaterial);
  bool propagate(const GlobalPoint& vertex, 
	         GlobalVector& Cand, int charge,
		 const MagneticField * bField) const;

private:
  bool   theIgnoreMaterial_;    /// whether or not propagation should ignore material
  double radius_, maxZ_, minZ_; /// Cylinder defining the inner surface of the calorimeter
};

#endif
