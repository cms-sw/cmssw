#ifndef volumeHandle_H
#define volumeHandle_H

/** \class MagGeoBuilderFromDDD::volumeHandle
 * A temporary container to cache info on a six-surface volume during
 * the processing. Used to sort, organise, and build shared planes.
 * One instance is created for each DDVolume. The parameters of the 
 * boundary surfaces are calculated during construction.
 *
 *  \author N. Amapane - INFN Torino (original developer)
 */

#include "BaseVolumeHandle.h"
#include "MagGeoBuilderFromDDD.h"

#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "MagneticField/VolumeGeometry/interface/VolumeSide.h"

class DDExpandedView;
class MagVolume6Faces;

class MagGeoBuilderFromDDD::volumeHandle : public magneticfield::BaseVolumeHandle {
public:
  volumeHandle(const DDExpandedView& fv, bool expand2Pi = false, bool debugVal = false);

  // Disallow Default/copy ctor & assignment op.
  // (we want to handle only pointers!!!)
  volumeHandle(const volumeHandle& v) = delete;
  volumeHandle operator=(const volumeHandle& v) = delete;

  DDSolidShape shape() const override { return solid.shape(); }

  /// The surfaces and they orientation, as required to build a MagVolume.
  std::vector<VolumeSide> sides() const override;

private:
  // initialise the refPlane
  void referencePlane(const DDExpandedView& fv);

  // Build the surfaces for a box
  void buildBox();
  // Build the surfaces for a trapezoid
  void buildTrap();
  // Build the surfaces for a ddtubs shape
  void buildTubs();
  // Build the surfaces for a ddcons shape
  void buildCons();
  // Build the surfaces for a ddpseudotrap shape
  void buildPseudoTrap();
  // Build the surfaces for a ddtrunctubs shape
  void buildTruncTubs();

  // the DDSolid.
  DDSolid solid;
};

#endif
