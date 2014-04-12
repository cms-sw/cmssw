#ifndef Geom_Bounds_H
#define Geom_Bounds_H


#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h" 

#include "DataFormats/GeometrySurface/interface/BoundSpan.h"

#include "FWCore/Utilities/interface/GCC11Compatibility.h"


/** \class Bounds 
 *
 *  Abstract base class for Bounds.
 *
 *  Bounds provide a general way to specify the form of a concrete 
 *  surface. For example, a BoundPlane with TrapezoidalPlaneBounds
 *  has a trapezoidal shape.
 */ 

class Bounds {
public:
  virtual ~Bounds() {}

  /// "Lenght" of the bounded volume; refer to the concrete class documentation
  /// for the specific definition.
  virtual float length() const = 0;

  /// "width" of the bounds; refer to the concrete class documentation
  /// for the specific definition.
  virtual float width() const = 0;

  /// "Thickness" of the bound around the surface;
  /// refer to the concrete class documentation for the specific definition.
  virtual float thickness() const = 0;

  /// Width at half length. Useful for e.g. pitch definition.
  virtual float widthAtHalfLength() const {return width();}

  /// Determine if the point is inside the bounds.
  virtual bool inside( const Local3DPoint&) const = 0;
  virtual bool inside( const Local2DPoint& p) const {
    return inside( Local3DPoint(p.x(), p.y(), 0) );
  }

  /// Determine if a point is inside the bounds, taking a tollerance into account
  virtual bool inside(const Local2DPoint& p, float tollerance) const {
    return inside(p,LocalError(tollerance*tollerance,0.f,tollerance*tollerance));
  }

  // For propagation with uncertainties - one has to know by how
  // much one missed a surface
  // virtual Local2DVector<float> howFar( const Local2DPoint&) = 0;
  //
  // Or maybe a better way of determining if a surface is "touched"
  // by a trajectory is to have a method

  /// Determine if a point is inside the bounds, taking error into account
  virtual bool inside( const Local3DPoint&, const LocalError&, float scale=1.f) const = 0;

  /// Determine if a 2D point is inside the bounds, taking error into account
  virtual bool inside( const Local2DPoint& p, const LocalError& err, 
		       float scale=1.f) const {
    return inside( Local3DPoint(p.x(), p.y(), 0), err, scale);
  }

  virtual Bounds* clone() const = 0;
  
  std::pair<float,float> const & phiSpan() const { return m_span.phiSpan(); }
  std::pair<float,float> const & zSpan()   const { return m_span.zSpan(); }
  std::pair<float,float> const & rSpan()   const { return m_span.rSpan(); }

  void computeSpan(Surface const & plane) { m_span.compute(plane);}

private:  
  
  BoundSpan m_span;

};

#endif
