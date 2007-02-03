// #include "Utilities/Configuration/interface/Architecture.h"

#include "MagneticField/VolumeGeometry/interface/MagVolume6Faces.h"

#ifndef NO_DETAILED_TIMING
// TOFIX
// #include "Utilities/Notification/interface/TimingReport.h"
//#include "Utilities/UI/interface/SimpleConfigurable.h"
#endif

MagVolume6Faces::MagVolume6Faces( const PositionType& pos,
				  const RotationType& rot, 
				  DDSolidShape shape,
				  const std::vector<VolumeSide>& faces,
				  const MagneticFieldProvider<float> * mfp)
  : MagVolume(pos,rot,shape,mfp), theFaces(faces) 
{
#ifndef NO_DETAILED_TIMING
// TOFIX
//   static SimpleConfigurable<bool> timerOn(false,"MagVolume6Faces:timing");
//   bool timerOn = false;
//   (*TimingReport::current()).switchOn("MagVolume6Faces::inside",timerOn);
#endif
}

bool MagVolume6Faces::inside( const GlobalPoint& gp, double tolerance) const 
{
#ifndef NO_DETAILED_TIMING
// TOFIX
//   static TimingReport::Item & timer = (*TimingReport::current())["MagVolume6Faces::inside"];
//   TimeMe t(timer,false);
#endif

  // check if the point is on the correct side of all delimiting surfaces
  for (std::vector<VolumeSide>::const_iterator i=theFaces.begin(); i!=theFaces.end(); ++i) {
    Surface::Side side = i->surface().side( gp, tolerance);
    if ( side != i->surfaceSide() && side != SurfaceOrientation::onSurface) return false;
  }
  return true;
}
