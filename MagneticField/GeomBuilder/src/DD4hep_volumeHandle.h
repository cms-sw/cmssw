#ifndef MagneticField_GeomBuilder_DD4hep_volumeHandle_h
#define MagneticField_GeomBuilder_DD4hep_volumeHandle_h

/** \class volumeHandle
 * A temporary container to cache info on a six-surface volume during
 * the processing. Used to sort, organise, and build shared planes.
 * One instance is created for each volume. The parameters of the 
 * boundary surfaces are calculated during construction.
 *
 *  \author N. Amapane - INFN Torino (original developer)
 */

#include "BaseVolumeHandle.h"

#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "MagneticField/VolumeGeometry/interface/VolumeSide.h"

namespace magneticfield {

  typedef const char* ShapeType;

  class volumeHandle : public BaseVolumeHandle {
  public:
    volumeHandle(const cms::DDFilteredView& fv, bool expand2Pi = false, bool debugVal = false);

    // Disallow Default/copy ctor & assignment op.
    // (we want to handle only pointers!!!)
    volumeHandle(const volumeHandle& v) = delete;
    volumeHandle operator=(const volumeHandle& v) = delete;

    // Shape at initialization
    DDSolidShape shape() const override { return (theShape); }

    /// The surfaces and they orientation, as required to build a MagVolume.
    std::vector<VolumeSide> sides() const override;

  private:
    // initialise the refPlane
    void referencePlane(const cms::DDFilteredView& fv);

    // Build the surfaces for a box
    void buildBox();
    // Build the surfaces for a trapezoid
    void buildTrap();
    // Build the surfaces for a ddtubs shape
    void buildTubs();
    // Build the surfaces for a ddcons shape
    void buildCons();
    // Build the surfaces for a ddtrunctubs shape
    void buildTruncTubs();

    // Shape at initialization
    const DDSolidShape theShape;
    const cms::DDFilteredView& solid;
    // "solid" name is for backwards compatibility. Can be changed to "fview" after DD4hep migration.
  };
}  // namespace magneticfield

#endif
