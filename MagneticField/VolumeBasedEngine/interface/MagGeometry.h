#ifndef MagGeometry_H
#define MagGeometry_H

/** \class MagGeometry
 *  Entry point to the geometry of magnetic volumes.
 *
 *  \author N. Amapane - INFN Torino
 */

#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "MagneticField/Layers/interface/MagBinFinders.h"

#include <vector>
#include <atomic>

class MagBLayer;
class MagESector;
class MagVolume;
class MagVolume6Faces;
template <class T>
class PeriodicBinFinderInPhi;

class MagGeometry {
public:
  typedef Surface::GlobalVector GlobalVector;
  typedef Surface::GlobalPoint GlobalPoint;

  /// Constructor
  MagGeometry(int geomVersion,
              const std::vector<MagBLayer*>&,
              const std::vector<MagESector*>&,
              const std::vector<MagVolume6Faces*>&,
              const std::vector<MagVolume6Faces*>&);
  MagGeometry(int geomVersion,
              const std::vector<MagBLayer const*>&,
              const std::vector<MagESector const*>&,
              const std::vector<MagVolume6Faces const*>&,
              const std::vector<MagVolume6Faces const*>&);

  /// Destructor
  ~MagGeometry();

  /// Return field vector at the specified global point
  GlobalVector fieldInTesla(const GlobalPoint& gp) const;

  /// Find a volume
  MagVolume const* findVolume(const GlobalPoint& gp, double tolerance = 0.) const;

  // FIXME: only for temporary tests, should be removed.
  const std::vector<MagVolume6Faces const*>& barrelVolumes() const { return theBVolumes; }
  const std::vector<MagVolume6Faces const*>& endcapVolumes() const { return theEVolumes; }

private:
  friend class MagGeometryExerciser;  // for debug purposes

  // Linear search (for debug purposes only)
  MagVolume const* findVolume1(const GlobalPoint& gp, double tolerance = 0.) const;

  bool inBarrel(const GlobalPoint& gp) const;

  const int me_;  // Instance ID, to trigger cache invalidation at IOV boundaries

  std::vector<MagBLayer const*> theBLayers;
  std::vector<MagESector const*> theESectors;

  // FIXME: only for temporary tests, should be removed.
  std::vector<MagVolume6Faces const*> theBVolumes;
  std::vector<MagVolume6Faces const*> theEVolumes;

  MagBinFinders::GeneralBinFinderInR<double> const* theBarrelBinFinder;
  PeriodicBinFinderInPhi<float> const* theEndcapBinFinder;

  bool cacheLastVolume;
  int geometryVersion;

  // boundaries of internal barrel-endcap volume separation
  double theBarrelRsq1;
  double theBarrelRsq2;
  double theBarrelZ0;
  double theBarrelZ1;
  double theBarrelZ2;
};
#endif
