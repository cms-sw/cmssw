#ifndef Alignment_CommonAlignment_AlignableSurface_H
#define Alignment_CommonAlignment_AlignableSurface_H

/** \class AlignableSurface
 *
 *  A class to hold a surface with width and length for alignment purposes.
 *
 *  $Date: 2007/04/25 18:37:59 $
 *  $Revision: 1.8 $
 *  \author Chung Khim Lae
 */

#include <vector>

#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"

class Plane;

class AlignableSurface:
  public GloballyPositioned<align::Scalar>
{
  public:

  /// Constructor to set surface from geometry.
  AlignableSurface(
		   const Plane& surface
		   );

  /// Constructor to set position and rotation; width and length default to 0.
  AlignableSurface(
		   const align::PositionType& = PositionType(), // default 0
		   const align::RotationType& = RotationType()  // default identity
		   );

  align::Scalar width() const { return theWidth; }

  align::Scalar length() const { return theLength; }

  void setWidth(align::Scalar width) { theWidth = width; }

  void setLength(align::Scalar length) { theLength = length; }

  using GloballyPositioned<align::Scalar>::toGlobal;
  using GloballyPositioned<align::Scalar>::toLocal;

  /// Return in global coord given a set of local points.
  align::GlobalPoints toGlobal(
			       const align::LocalPoints&
			       ) const;

  /// Return in global frame a rotation given in local frame.
  align::RotationType toGlobal(
			       const align::RotationType&
			       ) const;

  /// Return in global coord given Euler angles in local coord.
  align::EulerAngles toGlobal(
			      const align::EulerAngles&
			      ) const;

  /// Return in local frame a rotation given in global frame.
  align::RotationType toLocal(
			      const align::RotationType&
			      ) const;

  /// Return in local coord given Euler angles in global coord.
  align::EulerAngles toLocal(
			     const align::EulerAngles&
			     ) const;

  private:

  align::Scalar theWidth;
  align::Scalar theLength;
};

#endif
