#ifndef FourPointPlaneBounds_H
#define FourPointPlaneBounds_H

#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/GeometryVector/interface/Vector2DBase.h"

typedef Vector2DBase< float, LocalTag>    Local2DVector;

class FourPointPlaneBounds : public Bounds {
public:

    typedef LocalPoint::ScalarType    Scalar;

/** The corners are ASSUMED to come in cyclic order
 */
    FourPointPlaneBounds( const LocalPoint& a, const LocalPoint& b, 
			  const LocalPoint& c, const LocalPoint& d); 
    ~FourPointPlaneBounds() {}

  virtual float length() const;
  virtual float width() const;
  virtual float thickness() const;

    // basic bounds function
    virtual bool inside( const Local3DPoint& lp) const;
    
    virtual bool inside( const Local3DPoint& lp , const LocalError& e, float scale) const {
      return inside( lp);
    }

    virtual Bounds* clone() const {return new FourPointPlaneBounds(*this);}

private:

    Local2DPoint corners_[4];

    const Local2DPoint& corner(int i) const {return corners_[i%4];}
    double checkSide( int i, const Local2DPoint& lp) const;

    double checkSide( int i, Scalar x, Scalar y) const {
	const Local2DPoint& cor( corner(i));
	Local2DVector v( corner(i+1) - cor);
	// Local2DVector normal( -v.y(), v.x());  //  90 deg rotated
	return -v.y() * (x-cor.x()) + v.x() * (y-cor.y()); // == normal.dot(LP(x,y)-cor))
    }

};

#endif
