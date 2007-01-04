#ifndef GeometryVector_Geom_CoordinateSets_h
#define GeometryVector_Geom_CoordinateSets_h

#include <cmath>

namespace Geom {

/** Converts polar 2D coordinates to cartesian coordinates.
 *  Note: Spherical coordinates (also sometimes called Polar 3D coordinates)
 *  are handled by class Spherical2Cartesian
 */

    template <typename T>
    class Polar2Cartesian {
    public:
        /// Construct from radius and polar angle
	Polar2Cartesian( const T& r, const T& phi) :
	    r_(r), phi_(phi) {}

	const T& r() const   {return r_;}
	const T& phi() const {return phi_;}

	T x() const {return r_ * cos(phi_);}
	T y() const {return r_ * sin(phi_);}

    private:
	T r_;
	T phi_;
    };

/** Converts cylindtical coordinates to cartesian coordinates.
 */

    template <typename T>
    class Cylindrical2Cartesian {
    public:
        /** Construct from radius, azimuthal angle, and z component.
	 *  The radius in the cylindrical frame is the transverse component.
	 */
	Cylindrical2Cartesian( const T& r, const T& phi, const T& z) :
	    r_(r), phi_(phi), z_(z) {}

	const T& r() const   {return r_;}
	const T& phi() const {return phi_;}
	const T& z() const   {return z_;}

	T x() const {return r_ * cos(phi_);}
	T y() const {return r_ * sin(phi_);}

    private:
	T r_;
	T phi_;
	T z_;
    };

/** Converts spherical (or polar 3D) coordinates to cartesian coordinates.
 */

    template <typename T>
    class Spherical2Cartesian {
    public:
        /** Construct from polar angle, azimuthal angle, and radius.
	 *  The radius in the spherical frame is the magnitude of the vector.
	 */
	Spherical2Cartesian( const T& theta, const T& phi, const T& mag) :
	    theta_(theta), phi_(phi), r_(mag), 
	    transv_( sin(theta)*mag) {}

	const T& theta() const   {return theta_;}
	const T& phi() const     {return phi_;}
	const T& r() const       {return r_;}

	T x() const {return transv_ * cos(phi());}
	T y() const {return transv_ * sin(phi());}
	T z() const {return cos(theta()) * r();}

    private:
	T theta_;
	T phi_;
	T r_;
	T transv_;
    };


/** Cartesian coordinate set, for uniformity with other coordinate systems
 */

    template <typename T>
    class Cartesian2Cartesian3D {
    public:
	Cartesian2Cartesian3D( const T& x, const T& y, const T& z) :
	    x_(x), y_(y), z_(z) {}

	const T& x() const   {return x_;}
	const T& y() const   {return y_;}
	const T& z() const   {return z_;}
   private:
	T x_;
	T y_;
	T z_;
    };
}

#endif
