#ifndef RecoPixelVertexing_PixelTrackFitting_interface_CircleFromThreePoints_h
#define RecoPixelVertexing_PixelTrackFitting_interface_CircleFromThreePoints_h

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/Basic2DVector.h"

/** Computes the curvature (1/radius) and, if possible, the center
 *  of the circle passing through three points.
 *  The input points are three dimensional for convenience, but the
 *  calculation is done in the transverse (x,y) plane.
 *  No verification of the reasonableness of the z coordinate is done.
 *  If the three points lie on a line the curvature is zero and the center
 *  position is undefined. The 3 points are assumed to make sense:
 *  if the distance between two of them is very small compared to
 *  the ditance to the third the result will be numerically unstable.
 */

class CircleFromThreePoints {
public:

  /// dummy
  CircleFromThreePoints(){}

  typedef Basic2DVector<float>   Vector2D;

  /** Construct from three points (see class description).
   *  The order of points is not essential, but accuracy should be better if
   *  the second point lies between the other two on the circle.
   *  The optional argument "precision" specifies how accurately the
   *  straight line check has to be satisfied for setting the curvature
   *  to zero and the center position to "undefined".
   */
  CircleFromThreePoints( const GlobalPoint& inner,
			 const GlobalPoint& mid,
			 const GlobalPoint& outer,
			 double precision = 1.e-7);


  /** Returns the curvature (1/radius), in cm^(-1).
   *  The curvature is precomputed, this is just access method (takes no time).
   *  If curvature is zero the center is undefined
   *  (see description of presicion above).
   */
  float curvature() const { return theCurvature;}

  /** returns the position of the center of the circle.
   *  If curvature is zero, center() throws an exception to avoid use of
   *  undefined position.
   *  If the curvature is very small the position of the center will
   *  be very inaccurate.
   */
  Vector2D center() const { return theCenter; }

private:

  float    theCurvature;
  Vector2D theCenter;

  void init( const Vector2D& b, const Vector2D& c,
	     const Vector2D& offset, double precision);
};

#endif  // RecoPixelVertexing_PixelTrackFitting_interface_CircleFromThreePoints_h
