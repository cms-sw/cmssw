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
  class InterpolatorBuilder;

  class MagGeoBuilderFromDDD {
  public:
    /// Constructor.
    MagGeoBuilderFromDDD(std::string tableSet_,
                         int geometryVersion,
                         bool debug = false,
                         bool useMergeFileIfAvailable = true);

    /// Destructor
    ~MagGeoBuilderFromDDD();

    // Build the geometry.
    void build(const DDCompactView& cpv);

    ///  Set scaling factors for individual volumes.
    /// "keys" is a vector of 100*volume number + sector (sector 0 = all sectors)
    /// "values" are the corresponding scaling factors
    void setScaling(const std::vector<int>& keys, const std::vector<double>& values);

    void setGridFiles(const TableFileMap& gridFiles);

    /// Get barrel layers
    std::vector<MagBLayer*> barrelLayers() const;

    /// Get endcap layers
    std::vector<MagESector*> endcapSectors() const;

    float maxR() const;

    float maxZ() const;

    std::vector<MagVolume6Faces*> barrelVolumes() const;
    std::vector<MagVolume6Faces*> endcapVolumes() const;

    // Temporary container to manipulate volumes and their surfaces.
    class volumeHandle;  // Needs to be public to share code with DD4hep

  private:
    // Build interpolator for the volume with "correct" rotation
    MagProviderInterpol* buildInterpolator(const volumeHandle* vol, InterpolatorBuilder&) const;

    // Build all MagVolumes setting the MagProviderInterpol
    void buildMagVolumes(const handles& volumes, std::map<std::string, MagProviderInterpol*>& interpolators) const;

    // Print checksums for surfaces.
    void summary(handles& volumes) const;

    // Perform simple sanity checks
    void testInside(handles& volumes) const;

    handles bVolumes;  // the barrel volumes.
    handles eVolumes;  // the endcap volumes.

    std::vector<MagBLayer*> mBLayers;    // Finally built barrel geometry
    std::vector<MagESector*> mESectors;  // Finally built barrel geometry

    std::string tableSet;  // Version of the data files to be used
    int geometryVersion;   // Version of MF geometry

    std::map<int, double> theScalingFactors;
    const magneticfield::TableFileMap* theGridFiles;  // Non-owned pointer assumed to be valid until build() is called

    const bool debug;
    const bool useMergeFileIfAvailable;
  };
}  // namespace magneticfield
#endif
