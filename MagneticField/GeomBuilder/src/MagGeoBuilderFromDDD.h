#ifndef MagGeoBuilderFromDDD_H
#define MagGeoBuilderFromDDD_H

/** \class MagGeoBuilderFromDDD
 *  Parse the XML magnetic geometry, build individual volumes and match their
 *  shared surfaces. Build MagVolume6Faces and organise them in a hierarchical
 *  structure. Build MagGeometry out of it.
 *
 *  \author N. Amapane - INFN Torino
 */
#include "MagneticField/Interpolation/interface/MagProviderInterpol.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "CondFormats/MFObjects/interface/MagFieldConfig.h"

#include <string>
#include <vector>
#include <map>
#include <memory>

class Surface;
class MagBLayer;
class MagESector;
class MagVolume6Faces;
namespace magneticfield {
  class VolumeBasedMagneticFieldESProducer;
  class VolumeBasedMagneticFieldESProducerFromDB;
  class BaseVolumeHandle;  // Needs to be public to share code with DD4hep
  using handles = std::vector<BaseVolumeHandle*>;
}  // namespace magneticfield

class MagGeoBuilderFromDDD {
public:
  /// Constructor.
  MagGeoBuilderFromDDD(std::string tableSet_, int geometryVersion, bool debug = false);

  /// Destructor
  virtual ~MagGeoBuilderFromDDD();

  ///  Set scaling factors for individual volumes.
  /// "keys" is a vector of 100*volume number + sector (sector 0 = all sectors)
  /// "values" are the corresponding scaling factors
  void setScaling(const std::vector<int>& keys, const std::vector<double>& values);

  void setGridFiles(const magneticfield::TableFileMap& gridFiles);

  /// Get barrel layers
  std::vector<MagBLayer*> barrelLayers() const;

  /// Get endcap layers
  std::vector<MagESector*> endcapSectors() const;

  float maxR() const;

  float maxZ() const;

  // Temporary container to manipulate volumes and their surfaces.
  class volumeHandle;  // Needs to be public to share code with DD4hep

private:
  // Build the geometry.
  //virtual void build();
  virtual void build(const DDCompactView& cpv);

  // FIXME: only for temporary tests and debug, to be removed
  friend class TestMagVolume;
  friend class MagGeometry;
  friend class magneticfield::VolumeBasedMagneticFieldESProducer;
  friend class magneticfield::VolumeBasedMagneticFieldESProducerFromDB;

  std::vector<MagVolume6Faces*> barrelVolumes() const;
  std::vector<MagVolume6Faces*> endcapVolumes() const;

  // Build interpolator for the volume with "correct" rotation
  void buildInterpolator(const volumeHandle* vol, std::map<std::string, MagProviderInterpol*>& interpolators);

  // Build all MagVolumes setting the MagProviderInterpol
  void buildMagVolumes(const magneticfield::handles& volumes,
                       std::map<std::string, MagProviderInterpol*>& interpolators);

  // Print checksums for surfaces.
  void summary(magneticfield::handles& volumes);

  // Perform simple sanity checks
  void testInside(magneticfield::handles& volumes);

  magneticfield::handles bVolumes;  // the barrel volumes.
  magneticfield::handles eVolumes;  // the endcap volumes.

  std::vector<MagBLayer*> mBLayers;    // Finally built barrel geometry
  std::vector<MagESector*> mESectors;  // Finally built barrel geometry

  std::string tableSet;  // Version of the data files to be used
  int geometryVersion;   // Version of MF geometry

  std::map<int, double> theScalingFactors;
  const magneticfield::TableFileMap* theGridFiles;  // Non-owned pointer assumed to be valid until build() is called

  const bool debug;
};
#endif
