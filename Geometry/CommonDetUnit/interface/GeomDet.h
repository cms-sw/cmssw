#ifndef CommonDet_GeomDet_H
#define CommonDet_GeomDet_H

/** \class GeomDet
 *  Base class for GeomDetUnit and for composite GeomDet s. 
 *
 *  $Date: 2011/09/27 09:16:36 $
 *  $Revision: 1.13 $
 */


#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"

#include <vector>

class AlignmentPositionError;

class GeomDet {
public:
  typedef GeomDetEnumerators::SubDetector SubDetector;

  explicit GeomDet(BoundPlane* plane);

  explicit GeomDet(const ReferenceCountingPointer<BoundPlane>& plane);

  virtual ~GeomDet();

  /// The nominal surface of the GeomDet
  virtual const BoundPlane& surface() const {return *thePlane;}

  /// Same as surface(), kept for backward compatibility
  virtual const BoundPlane& specificSurface() const {return *thePlane;}
  
  /// The position (origin of the R.F.)
  const Surface::PositionType& position() const {return surface().position();} 
  
  /// The rotation defining the local R.F.
  const Surface::RotationType& rotation() const { return surface().rotation();}

  /// Conversion to the global R.F. from the R.F. of the GeomDet
  GlobalPoint toGlobal(const Local2DPoint& lp) const {
    return surface().toGlobal( lp);
  }
  
  /// Conversion to the global R.F. from the R.F. of the GeomDet
  GlobalPoint toGlobal(const Local3DPoint& lp) const {
    return surface().toGlobal( lp);
  }

  /// Conversion to the global R.F. from the R.F. of the GeomDet
  GlobalVector toGlobal(const LocalVector& lv) const {
    return surface().toGlobal( lv);
  }
  
  /// Conversion to the R.F. of the GeomDet
  LocalPoint toLocal(const GlobalPoint& gp) const {
    return surface().toLocal( gp);
  }
  
  /// Conversion to the R.F. of the GeomDet
  LocalVector toLocal(const GlobalVector& gv) const {
    return surface().toLocal( gv);
  } 

  /// The label of this GeomDet
  DetId geographicalId() const { return m_detId; }

  /// Which subdetector
  virtual SubDetector subDetector() const = 0;  

  /// Return pointer to alignment errors. 
  /// Defaults to "null" if not reimplemented in the derived classes.
  AlignmentPositionError* alignmentPositionError() const { return theAlignmentPositionError;}
  LocalError const & localError() const { return theLocalError;}

  /// Returns direct components, if any
  virtual std::vector< const GeomDet*> components() const = 0;

  /// Returns a component GeomDet given its DetId, if existing
  // FIXME: must become pure virtual
  virtual const GeomDet* component(DetId /*id*/) const {return 0;}


  protected:

    void setDetId(DetId id) {
      m_detId = id;
    }

private:

  ReferenceCountingPointer<BoundPlane>  thePlane;
  AlignmentPositionError*               theAlignmentPositionError;
  LocalError                            theLocalError;
  DetId m_detId;


  /// Alignment part of interface, available only to friend 
  friend class DetPositioner;

  /// Relative displacement (with respect to current position).
  /// Does not move components (if any).
  void move( const GlobalVector& displacement);

  /// Relative rotation (with respect to current orientation).
  /// Does not move components (if any).
  void rotate( const Surface::RotationType& rotation);

  /// Replaces the current position and rotation with new ones.
  /// actually replaces the surface with a new surface.
  /// Does not move components (if any).
   
  void setPosition( const Surface::PositionType& position, 
		    const Surface::RotationType& rotation);

  /// create the AlignmentPositionError for this Det if not existing yet,
  /// or replace the existing one by the given one. For adding, use the
  /// +=,-=  methods of the AlignmentPositionError
  /// Does not affect the AlignmentPositionError of components (if any).
  
  void setAlignmentPositionError (const AlignmentPositionError& ape); 

};
  
#endif




