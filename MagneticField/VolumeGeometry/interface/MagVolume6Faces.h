#ifndef MagVolume6Faces_h
#define MagVolume6Faces_h

/** \class MagVolume6Faces
 *
 *  A MagVolume defined by a number of sides (surfaces)
 *  NOTE that despite the name the sides can be less (or more) than 6!!! <br>
 *
 *  inside() is implemented by checking that the given point is on the 
 *  correct side of each of the surfaces sides.
 *
 *  \author T. Todorov, N. Amapane
 */

#include "MagneticField/VolumeGeometry/interface/MagVolume.h"
#include "MagneticField/VolumeGeometry/interface/VolumeSide.h"

#include <vector>

template <class T>
class MagneticFieldProvider;

class MagVolume6Faces : public MagVolume {
public:

  MagVolume6Faces( const PositionType& pos, const RotationType& rot, 
		   const std::vector<VolumeSide>& faces,
		   const MagneticFieldProvider<float> * mfp,
		   double sf=1.);

  using MagVolume::inside;
  bool inside( const GlobalPoint& gp, double tolerance=0.) const override;

  /// Access to volume faces
  const std::vector<VolumeSide>& faces() const override {return theFaces;}

  //--> These are used for debugging purposes only
  short volumeNo;
  char copyno;
  //<--

private:

  std::vector<VolumeSide> theFaces;

};

#endif
