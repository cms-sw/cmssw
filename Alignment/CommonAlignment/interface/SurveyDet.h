#ifndef Alignment_SurveyAnalysis_SurveyDet_h
#define Alignment_SurveyAnalysis_SurveyDet_h

/** \class SurveyDet
 *
 *  Class to hold survey info.
 *
 *  $Date: 2007/01/17 $
 *  $Revision: 1 $
 *  \author Chung Khim Lae
 */

#include "Alignment/CommonAlignment/interface/AlignableSurface.h"
#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/LocalPoint.h"

class SurveyDet
{
  public:

  /// Set the surface and 9 survey points to find its position and rotation
  ///
  ///  -----------   ^   ----------- 
  /// | . | . | . |  |  | 4 | 3 | 2 |
  ///  -----------   |    ----------- 
  /// | . | . | . |  L  | 5 | 0 | 1 |
  ///  -----------   |   ----------- 
  /// | . | . | . |  |  | 6 | 7 | 8 |
  ///  -----------   v   ----------- 
  /// <---- W ---->
  ///
  /// The left sensor shows how the 9 points are chosen (W = width, L = length)
  /// The right sensor shows how the points are indexed
  SurveyDet(
	    const AlignableSurface& // set the surface
	    );

  inline const AlignableSurface::PositionType& position() const;

  inline const AlignableSurface::RotationType& rotation() const;

  inline const std::vector<LocalPoint>& localPoints() const;

  inline std::vector<GlobalPoint> globalPoints() const;

  /// Find the Jacobian for a local point to be used in HIP algo.
  /// Does not check the range of index of local point.
  AlgebraicMatrix derivatives(
			      unsigned int index // index of point
			      ) const;

  private:

  AlignableSurface theSurface; // surface of det from survey info

  std::vector<LocalPoint> thePoints; // survey points on the surface
};

const AlignableSurface::PositionType& SurveyDet::position() const
{
  return theSurface.position();
}

const AlignableSurface::RotationType& SurveyDet::rotation() const
{
  return theSurface.rotation();
}

const std::vector<LocalPoint>& SurveyDet::localPoints() const
{
  return thePoints;
}

std::vector<GlobalPoint> SurveyDet::globalPoints() const
{
  return theSurface.toGlobal(thePoints);
}

#endif
