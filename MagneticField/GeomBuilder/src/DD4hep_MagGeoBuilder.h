#ifndef MagneticField_GeomBuilder_DD4hep_MagGeoBuilder_h
#define MagneticField_GeomBuilder_DD4hep_MagGeoBuilder_h

/** \class MagGeoBuilder
 *  Parse the XML magnetic geometry, build individual volumes and match their
 *  shared surfaces. Build MagVolume6Faces and organise them in a hierarchical
 *  structure. Build MagGeometry out of it.
 *
 *  \author N. Amapane - INFN Torino (original developer)
 */

#include "CondFormats/MFObjects/interface/MagFieldConfig.h"
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "MagneticField/Interpolation/interface/MagProviderInterpol.h"
#include "DD4hep_volumeHandle.h"

#include <string>
#include <vector>
#include <map>
#include <memory>

namespace dd4hep {
  class Detector;
}

class Surface;
class MagBLayer;
class MagESector;
class MagVolume6Faces;

namespace magneticfield {
  class InterpolatorBuilder;

  class MagGeoBuilder {
  public:
    MagGeoBuilder(std::string tableSet, int geometryVersion, bool debug = false, bool useMergeFileIfAvailable = true);

    ~MagGeoBuilder();

    // Build the geometry.
    void build(const cms::DDDetector* det);

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

  private:
    // Build interpolator for the volume with "correct" rotation
    MagProviderInterpol* buildInterpolator(const volumeHandle* vol, InterpolatorBuilder&) const;

    // Build all MagVolumes setting the MagProviderInterpol
    void buildMagVolumes(const handles& volumes,
                         const std::map<std::string, MagProviderInterpol*>& interpolators) const;

    // Print checksums for surfaces.
    void summary(handles& volumes) const;

    // Perform simple sanity checks
    void testInside(handles& volumes) const;

    handles bVolumes_;  // the barrel volumes.
    handles eVolumes_;  // the endcap volumes.

    std::vector<MagBLayer*> mBLayers_;    // Finally built barrel geometry
    std::vector<MagESector*> mESectors_;  // Finally built barrel geometry

    std::string tableSet_;  // Version of the data files to be used
    int geometryVersion_;   // Version of MF geometry

    std::map<int, double> theScalingFactors_;
    const TableFileMap* theGridFiles_;  // Non-owned pointer assumed to be valid until build() is called

    const bool debug_;
    const bool useMergeFileIfAvailable_;
  };
}  // namespace magneticfield
#endif
