#ifndef CommonDet_GeomDet_H
#define CommonDet_GeomDet_H

/** \class GeomDet
 *  Base class for GeomDetUnit and for composite GeomDet s. 
 *
 */

#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "Geometry/CommonTopologies/interface/GeomDetEnumerators.h"

#include <vector>

class AlignmentPositionError;

class Topology;
class GeomDetType;
class SurfaceDeformation;

class GeomDet {
public:
  using SubDetector = GeomDetEnumerators::SubDetector;

  explicit GeomDet(Plane* plane) : thePlane(plane) {}
  explicit GeomDet(const ReferenceCountingPointer<Plane>& plane) : thePlane(plane) {}

  virtual ~GeomDet();

  /// The nominal surface of the GeomDet
  const Plane& surface() const { return *thePlane; }

  /// Same as surface(), kept for backward compatibility
  const Plane& specificSurface() const { return *thePlane; }

  /// The position (origin of the R.F.)
  const Surface::PositionType& position() const { return surface().position(); }

  /// The rotation defining the local R.F.
  const Surface::RotationType& rotation() const { return surface().rotation(); }

  /// Conversion to the global R.F. from the R.F. of the GeomDet
  GlobalPoint toGlobal(const Local2DPoint& lp) const { return surface().toGlobal(lp); }

  /// Conversion to the global R.F. from the R.F. of the GeomDet
  GlobalPoint toGlobal(const Local3DPoint& lp) const { return surface().toGlobal(lp); }

  /// Conversion to the global R.F. from the R.F. of the GeomDet
  GlobalVector toGlobal(const LocalVector& lv) const { return surface().toGlobal(lv); }

  /// Conversion to the R.F. of the GeomDet
  LocalPoint toLocal(const GlobalPoint& gp) const { return surface().toLocal(gp); }

  /// Conversion to the R.F. of the GeomDet
  LocalVector toLocal(const GlobalVector& gv) const { return surface().toLocal(gv); }

  /// The label of this GeomDet
  DetId geographicalId() const { return m_detId; }

  /// Which subdetector
  virtual SubDetector subDetector() const;

  /// is a Unit
  virtual bool isLeaf() const { return components().empty(); }

  /// Returns direct components, if any
  virtual std::vector<const GeomDet*> components() const { return std::vector<const GeomDet*>(); }

  /// Returns a component GeomDet given its DetId, if existing
  // FIXME: must become pure virtual
  virtual const GeomDet* component(DetId /*id*/) const { return nullptr; }

  /// Return pointer to alignment errors.
  AlignmentPositionError const* alignmentPositionError() const { return theAlignmentPositionError; }

  // specific unit index in a given subdetector (such as Tracker)
  int index() const { return m_index; }
  void setIndex(int i) { m_index = i; }

  // specific geomDet index in a given subdetector (such as Tracker)
  int gdetIndex() const { return m_gdetIndex; }
  void setGdetIndex(int i) { m_gdetIndex = i; }

  virtual const Topology& topology() const;

  virtual const GeomDetType& type() const;

  /// Return pointer to surface deformation.
  /// Defaults to "null" if not reimplemented in the derived classes.
  virtual const SurfaceDeformation* surfaceDeformation() const { return nullptr; }

protected:
  void setDetId(DetId id) { m_detId = id; }

private:
  ReferenceCountingPointer<Plane> thePlane;
  DetId m_detId;
  int m_index = -1;
  int m_gdetIndex = -1;
  AlignmentPositionError* theAlignmentPositionError = nullptr;

protected:
  /// set the LocalAlignmentError properly trasforming the ape
  /// Does not affect the AlignmentPositionError of components (if any).
  virtual bool setAlignmentPositionError(const AlignmentPositionError& ape);

private:
  /// Alignment part of interface, available only to friend
  friend class DetPositioner;

  /// Relative displacement (with respect to current position).
  /// Does not move components (if any).
  void move(const GlobalVector& displacement);

  /// Relative rotation (with respect to current orientation).
  /// Does not move components (if any).
  void rotate(const Surface::RotationType& rotation);

  /// Replaces the current position and rotation with new ones.
  /// actually replaces the surface with a new surface.
  /// Does not move components (if any).

  void setPosition(const Surface::PositionType& position, const Surface::RotationType& rotation);

private:
  /// set the SurfaceDeformation for this GeomDetUnit.
  /// Does not affect the SurfaceDeformation of components (if any).
  /// Throws if not implemented in derived class.
  virtual void setSurfaceDeformation(const SurfaceDeformation* deformation);
};

using GeomDetUnit = GeomDet;

#endif
