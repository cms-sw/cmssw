#ifndef Alignment_CommonAlignment_AlignableSurface_H
#define Alignment_CommonAlignment_AlignableSurface_H

/** \class AlignableSurface
 *
 *  A class to hold a surface with width and length for alignment purposes.
 *
 *  $Date: 2007/02/22 01:53:40 $
 *  $Revision: 1.3 $
 *  \author Chung Khim Lae
 */

#include <vector>

#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"

class AlignableSurface:
  public GloballyPositioned<float>
{
  public:

  /// Constructor to set position and rotation; width and length default to 0
  AlignableSurface(
		   const PositionType& = PositionType(), // default 0
		   const RotationType& = RotationType()  // default identity
		   );

  float width() const { return theWidth; }

  float length() const { return theLength; }

  void setWidth(float width) { theWidth = width; }

  void setLength(float length) { theLength = length; }

  using GloballyPositioned<float>::toGlobal;
  using GloballyPositioned<float>::toLocal;

  /// Return in global coord given a set of local points.
  std::vector<GlobalPoint> toGlobal(
				    const std::vector<LocalPoint>&
				    ) const;

  /// Return in local frame a rotation given in global frame.
  TkRotation<float> toLocal(
			    const TkRotation<float>&
			    ) const;

  private:

  float theWidth;
  float theLength;
};

#endif
