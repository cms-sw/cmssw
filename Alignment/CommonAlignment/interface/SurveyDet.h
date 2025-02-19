#ifndef Alignment_CommonAlignment_SurveyDet_h
#define Alignment_CommonAlignment_SurveyDet_h

/** \class SurveyDet
 *
 *  Class to hold survey info.
 *
 *  $Date: 2007/10/08 13:21:29 $
 *  $Revision: 1.5 $
 *  \author Chung Khim Lae
 */

#include "Alignment/CommonAlignment/interface/AlignableSurface.h"

class SurveyDet
{
  public:

  /// Set the surface and 9 survey points to find its position and rotation.
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
  /// The right sensor shows how the points are indexed.
  /// Also set the survey errors.
  SurveyDet(
	    const AlignableSurface&,  // set the surface
	    const align::ErrorMatrix& // set the survey errors
	    );

  inline const align::PositionType& position() const;

  inline const align::RotationType& rotation() const;

  inline const align::ErrorMatrix& errors() const;

  inline const align::LocalPoints& localPoints() const;

  inline align::GlobalPoints globalPoints() const;

  /// Find the Jacobian for a local point to be used in HIP algo.
  /// Does not check the range of index of local point.
  AlgebraicMatrix derivatives(
			      unsigned int index // index of point
			      ) const;

  private:

  AlignableSurface theSurface; // surface of det from survey info

  align::ErrorMatrix theErrors;

  std::vector<align::LocalPoint> thePoints; // survey points on the surface
};

const align::PositionType& SurveyDet::position() const
{
  return theSurface.position();
}

const align::RotationType& SurveyDet::rotation() const
{
  return theSurface.rotation();
}

const align::ErrorMatrix& SurveyDet::errors() const
{
  return theErrors;
}

const align::LocalPoints& SurveyDet::localPoints() const
{
  return thePoints;
}

align::GlobalPoints SurveyDet::globalPoints() const
{
  return theSurface.toGlobal(thePoints);
}

#endif
