#ifndef Alignment_CommonAlignment_AlignableSurface_H
#define Alignment_CommonAlignment_AlignableSurface_H

/** \class AlignableSurface
 *
 *  A class to hold a surface with width and length for alignment purposes.
 *
 *  $Date: 2007/01/28 $
 *  $Revision: 1 $
 *  \author Chung Khim Lae
 */

#include <vector>

#include "Geometry/Surface/interface/GloballyPositioned.h"

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

  /// Return in global coord given a set of local points.
  std::vector<GlobalPoint> toGlobal(
				    const std::vector<LocalPoint>&
				    ) const;

  private:

  float theWidth;
  float theLength;
};

#endif
