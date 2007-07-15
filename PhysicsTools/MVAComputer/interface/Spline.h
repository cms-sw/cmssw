#ifndef PhysicsTools_MVAComputer_Spline_h
#define PhysicsTools_MVAComputer_Spline_h
// -*- C++ -*-
//
// Package:     MVAComputer
// Class  :     Spline
//

//
// Author:	Christophe Saout <christophe.saout@cern.ch>
// Created:     Sat Apr 24 15:18 CEST 2007
// $Id: Spline.h,v 1.2 2007/05/25 16:37:58 saout Exp $
//

namespace PhysicsTools {

/** \class Spline
 *
 * \short A simple class for cubic splines
 *
 * This class implements cubic splines for n equidistant points in x
 * between 0 and 1. It is constructed from an array of n y coordinates
 * and can compute the interpolated y coordinate for a given x.
 *
 ************************************************************/
class Spline {
    public:
	Spline();
	Spline(const Spline &orig);

	/// construct spline from \a n y coordinates in array \a vals
	Spline(unsigned int n, const double *vals);
	~Spline();

	Spline &operator = (const Spline &orig);

	/// initialize spline from \a n y coordinates in array \a vals
	void set(unsigned int n, const double *vals);

	/// compute y coordinate at x coordinate \a x
	double eval(double x) const;

	/// compute integral under curve between 0 and \a x
	double integral(double x) const;

	/// total area (integral between 0 and 1) under curve
	double getArea() const { return area; }

    private:
	/// internal class describing a "segment" (between two x points)
	struct Segment {
		double coeffs[4];
		double area;

		double eval(double x) const;
		double integral(double x) const;
	};

	unsigned int	n;
	Segment		*segments;
	double		area;
};

} // namespace PhysicsTools

#endif // PhysicsTools_MVAComputer_Spline_h
