#ifndef MagGeometry_H
#define MagGeometry_H

/** \class MagGeometry
 *  Entry point to the geometry of magnetic volumes.
 *
 *  $Date: 2004/06/22 17:05:10 $
 *  $Revision: 1.5 $
 *  \author N. Amapane - INFN Torino
 */

#include "Geometry/Surface/interface/BoundPlane.h"
#include "MagneticField/Layers/src/MagBinFinders.h"
/* #include "Utilities/Notification/interface/Singleton.h" */
#include "DetectorDescription/Base/interface/Singleton.h"
#include <vector>

class MagBLayer;
class MagESector;
class MagVolume;
class MagVolume6Faces;
template <class T> class PeriodicBinFinderInPhi;

class MagGeometry {
public:

  typedef Surface::GlobalVector   GlobalVector;
  typedef Surface::GlobalPoint    GlobalPoint;

  /// Destructor
  ~MagGeometry();

  /// Return field vector at the specified global point
  GlobalVector fieldInTesla(const GlobalPoint & gp) const;

  /// Find a volume
  MagVolume * findVolume(const GlobalPoint & gp) const;

  // FIXME: only for temporary tests, should be removed.
  const std::vector<MagVolume6Faces*> & barrelVolumes() {return theBVolumes;}
  const std::vector<MagVolume6Faces*> & endcapVolumes() {return theEVolumes;}

private:
  /// Constructor; allow construction from a Singleton only.
  MagGeometry();
  friend class DDI::Singleton<MagGeometry>;

  friend class MagGeometryExerciser; // for debug purposes

  // Linear search (for debug purposes only)
  MagVolume * findVolume1(const GlobalPoint & gp, double tolerance=0.) const;

  // Use hierarchical structure for the barrel with non-optimized R search
  MagVolume * findVolume2(const GlobalPoint & gp, double tolerance=0.) const;

  bool inBarrel(const GlobalPoint& gp) const;

  mutable MagVolume * lastVolume; // Cache last volume found

  std::vector<MagBLayer *> theBLayers;
  std::vector<MagESector *> theESectors;

  // FIXME: only for temporary tests, should be removed.
  std::vector<MagVolume6Faces*> theBVolumes;
  std::vector<MagVolume6Faces*> theEVolumes;

  MagBinFinders::GeneralBinFinderInR<double>* theBarrelBinFinder;
  PeriodicBinFinderInPhi<float> * theEndcapBinFinder;

};
#endif

