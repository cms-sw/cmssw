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
	float eval(float x) const;

	/// compute the derivate at x coordinate \a x
	float deriv(float x) const;

	/// compute integral under curve between 0 and \a x
	float integral(float x) const;

	/// total area (integral between 0 and 1) under curve
	float getArea() const { return area; }

	/// return the number of entries
	inline unsigned int numberOfEntries() const { return n + 1; }

    private:
	/// internal class describing a "segment" (between two x points)
	struct Segment {
		float coeffs[4];
		float area;

		float eval(float x) const;
		float deriv(float x) const;
		float integral(float x) const;
	};

	unsigned int	n;
	Segment		*segments;
	float		area;
};

} // namespace PhysicsTools

#endif // PhysicsTools_MVAComputer_Spline_h
