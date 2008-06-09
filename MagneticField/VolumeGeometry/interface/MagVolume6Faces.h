#ifndef MagVolume6Faces_H
#define MagVolume6Faces_H

#include "MagneticField/VolumeGeometry/interface/MagVolume.h"
#include "MagneticField/VolumeGeometry/interface/VolumeSide.h"

#include <vector>

//-- FIXME
#include <string>
//--

template <class T>
class MagneticFieldProvider;

class MagVolume6Faces : public MagVolume {
public:

  MagVolume6Faces( const PositionType& pos, const RotationType& rot, 
		   DDSolidShape shape, const std::vector<VolumeSide>& faces,
		   const MagneticFieldProvider<float> * mfp);

  using MagVolume::inside;
  virtual bool inside( const GlobalPoint& gp, double tolerance=0.) const;

  /// Access to volume faces
  std::vector<VolumeSide> faces() const {return theFaces;}

  //-- FIXME
  std::string name;
  //--

private:

  std::vector<VolumeSide> theFaces;

};

#endif
