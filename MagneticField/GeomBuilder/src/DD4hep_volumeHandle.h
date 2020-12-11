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
    void buildBox(double halfX, double halfY, double halfZ);
    // Build the surfaces for a trapezoid
    void buildTrap(double x1,
                   double x2,
                   double x3,
                   double x4,
                   double y1,
                   double y2,
                   double theta,
                   double phi,
                   double halfZ,
                   double alpha1,
                   double alpha2);
    // Build the surfaces for a ddtubs shape
    void buildTubs(double zhalf, double rIn, double rOut, double startPhi, double deltaPhi);
    // Build the surfaces for a ddcons shape
    void buildCons(double zhalf,
                   double rInMinusZ,
                   double rOutMinusZ,
                   double rInPlusZ,
                   double rOutPlusZ,
                   double startPhi,
                   double deltaPhi);
    // Build the surfaces for a ddpseudotrap. This is not a supported
    // shape in DD4hep; it is handled here to cope with legacy geometries.
    void buildPseudoTrap(double x1, double x2, double y1, double y2, double halfZ, double radius, bool atMinusZ);
    // Build the surfaces for a ddtrunctubs shape
    void buildTruncTubs(double zhalf,
                        double rIn,
                        double rOut,
                        double startPhi,
                        double deltaPhi,
                        double cutAtStart,
                        double cutAtDelta,
                        bool cutInside);

    // Shape at initialization
    const DDSolidShape theShape;
    const cms::DDFilteredView& solid;
    // "solid" name is for backwards compatibility. Can be changed to "fview" after DD4hep migration.
  };
}  // namespace magneticfield

#endif
