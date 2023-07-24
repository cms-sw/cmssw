#ifndef Geometry_TrackerTopology_ProxyStripTopology_H
#define Geometry_TrackerTopology_ProxyStripTopology_H

/// ProxyStripTopology
///
/// Class derived from StripTopology that serves as a proxy to the
/// actual topology for a given StripGeomDetType. In addition, the
/// class holds a pointer to the surface deformation parameters.
/// ProxyStripTopology takes over ownership of the surface
/// deformation parameters.
///
/// All inherited virtual methods that take the
/// predicted track state as a parameter are reimplemented in order
/// to apply corrections due to the surface deformations.
//
/// The 'old' methods without the track predictions simply call
/// the method of the actual StripTopology.
/// While one could easily deduce corrections from the given
/// LocalPosition (and track angles 0) when converting from local frame
/// to measurement frame, this is not done to be consistent with the
/// methods converting the other way round where the essential y-coordinate
/// is basically missing (it is a 1D strip detector...)
///
///  \author    : Andreas Mussgiller
///  date       : November 2010

#include "Geometry/CommonTopologies/interface/SurfaceDeformation.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include <memory>

class Plane;

class ProxyStripTopology final : public StripTopology {
public:
  ProxyStripTopology(StripGeomDetType const* type, Plane* bp);

  LocalPoint localPosition(const MeasurementPoint& mp) const override { return specificTopology().localPosition(mp); }
  /// conversion taking also the predicted track state
  LocalPoint localPosition(const MeasurementPoint& mp, const Topology::LocalTrackPred& trkPred) const override;

  LocalPoint localPosition(float strip) const override { return specificTopology().localPosition(strip); }
  /// conversion taking also the predicted track state
  LocalPoint localPosition(float strip, const Topology::LocalTrackPred& trkPred) const override;

  LocalError localError(float strip, float stripErr2) const override {
    return specificTopology().localError(strip, stripErr2);
  }
  /// conversion taking also the predicted track state
  LocalError localError(float strip, float stripErr2, const Topology::LocalTrackPred& trkPred) const override;

  LocalError localError(const MeasurementPoint& mp, const MeasurementError& me) const override {
    return specificTopology().localError(mp, me);
  }
  /// conversion taking also the predicted track state
  LocalError localError(const MeasurementPoint& mp,
                        const MeasurementError& me,
                        const Topology::LocalTrackPred& trkPred) const override;

  MeasurementPoint measurementPosition(const LocalPoint& lp) const override {
    return specificTopology().measurementPosition(lp);
  }
  MeasurementPoint measurementPosition(const LocalPoint& lp, const Topology::LocalTrackAngles& dir) const override;

  MeasurementError measurementError(const LocalPoint& lp, const LocalError& le) const override {
    return specificTopology().measurementError(lp, le);
  }
  MeasurementError measurementError(const LocalPoint& lp,
                                    const LocalError& le,
                                    const Topology::LocalTrackAngles& dir) const override;

  int channel(const LocalPoint& lp) const override { return specificTopology().channel(lp); }
  int channel(const LocalPoint& lp, const Topology::LocalTrackAngles& dir) const override;

  float strip(const LocalPoint& lp) const override { return specificTopology().strip(lp); }
  /// conversion taking also the track state (LocalTrajectoryParameters)
  float strip(const LocalPoint& lp, const Topology::LocalTrackAngles& dir) const override;

  float coveredStrips(const LocalPoint& lp1, const LocalPoint& lp2) const override {
    return specificTopology().coveredStrips(lp1, lp2);
  }

  float pitch() const override { return specificTopology().pitch(); }
  float localPitch(const LocalPoint& lp) const override { return specificTopology().localPitch(lp); }
  /// conversion taking also the angle from the track state (LocalTrajectoryParameters)
  float localPitch(const LocalPoint& lp, const Topology::LocalTrackAngles& dir) const override;

  float stripAngle(float strip) const override { return specificTopology().stripAngle(strip); }

  int nstrips() const override { return specificTopology().nstrips(); }

  float stripLength() const override { return specificTopology().stripLength(); }
  float localStripLength(const LocalPoint& lp) const override { return specificTopology().localStripLength(lp); }
  float localStripLength(const LocalPoint& lp, const Topology::LocalTrackAngles& dir) const override;

  virtual const GeomDetType& type() const { return *theType; }
  virtual StripGeomDetType const& specificType() const { return *theType; }

  const SurfaceDeformation* surfaceDeformation() const { return theSurfaceDeformation.operator->(); }
  virtual void setSurfaceDeformation(const SurfaceDeformation* deformation);

  virtual const StripTopology& specificTopology() const { return specificType().specificTopology(); }

private:
  /// Internal method to get correction of the position from SurfaceDeformation,
  /// must not be called if 'theSurfaceDeformation' is a null pointer.
  SurfaceDeformation::Local2DVector positionCorrection(const LocalPoint& pos,
                                                       const Topology::LocalTrackAngles& dir) const;
  /// Internal method to get correction of the position from SurfaceDeformation,
  /// must not be called if 'theSurfaceDeformation' is a null pointer.
  SurfaceDeformation::Local2DVector positionCorrection(const Topology::LocalTrackPred& trk) const;

  StripGeomDetType const* theType;
  float theLength, theWidth;
  std::unique_ptr<const SurfaceDeformation> theSurfaceDeformation;
};

#endif
