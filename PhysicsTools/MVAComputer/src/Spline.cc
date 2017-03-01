// -*- C++ -*-
//
// Package:     MVAComputer
// Class  :     Spline
// 

// Implementation:
//     Simple cubic spline implementation for equidistant points in x.
//
// Author:      Christophe Saout
// Created:     Sat Apr 24 15:18 CEST 2007
//

#include <cstring>
#include <cmath>

#include "PhysicsTools/MVAComputer/interface/Spline.h"

namespace PhysicsTools {

float Spline::Segment::eval(float x) const
{
	float tmp;
	float y = 0.0;
	y += coeffs[0];		tmp = x;
	y += coeffs[1] * tmp;	tmp *= x;
	y += coeffs[2] * tmp;	tmp *= x;
	y += coeffs[3] * tmp;
	return y;
}

float Spline::Segment::deriv(float x) const
{
	float tmp;
	float d = 0.0;
	d += coeffs[1];			tmp = x;
	d += coeffs[2] * tmp * 2.0;	tmp *= x;
	d += coeffs[3] * tmp * 3.0;
	return d;
}

float Spline::Segment::integral(float x) const
{
	float tmp = x;
	float area = this->area;
	area += coeffs[0] * tmp;	       tmp *= x;
	area += coeffs[1] * tmp * (1.0 / 2.0); tmp *= x;
	area += coeffs[2] * tmp * (1.0 / 3.0); tmp *= x;
	area += coeffs[3] * tmp * (1.0 / 4.0);
	return area;
}

Spline::Spline() : n(0), segments(0), area(0.0)
{}

Spline::Spline(const Spline &orig) : n(orig.n), area(orig.area)
{
	segments = new Segment[n];
	std::memcpy(segments, orig.segments, sizeof(Segment) * n);
}

Spline::Spline(unsigned int n_, const double *vals) :
	n(0), segments(0), area(0.0)
{ set(n_, vals); }

void Spline::set(unsigned int n_, const double *vals)
{
	n = n_ - 1;
	area = 0.0;

	delete[] segments;
	segments = new Segment[n];

	if (n == 1) {
		Segment *seg = &segments[0];
		seg->coeffs[0] = vals[0];
		seg->coeffs[1] = vals[1] - vals[0];
		seg->coeffs[2] = 0.0;
		seg->coeffs[3] = 0.0;
		seg->area = 0.0;
		area = seg->integral(1.0);
		return;
	}

	float m0, m1;
	Segment *seg = &segments[0];
	m0 = 0.0, m1 = 0.5 * (vals[2] - vals[0]);
	seg->coeffs[0] = vals[0];
	seg->coeffs[1] = -2.0 * vals[0] + 2.0 * vals[1] - m1;
	seg->coeffs[2] = vals[0] - vals[1] + m1;
	seg->coeffs[3] = 0.0;
	seg->area = 0.0;
	area = seg->integral(1.0);
	m0 = m1;
	seg++, vals++;

	for(unsigned int i = 1; i < n - 1; i++, seg++, vals++) {
		m1 = 0.5 * (vals[2] - vals[0]);
		seg->coeffs[0] = vals[0];
		seg->coeffs[1] = m0;
		seg->coeffs[2] = -3.0 * vals[0] - 2.0 * m0 + 3.0 * vals[1] - m1;
		seg->coeffs[3] = 2.0 * vals[0] + m0 - 2.0 * vals[1] + m1;
		seg->area = area;
		area = seg->integral(1.0);
		m0 = m1;
	}

	seg->coeffs[0] = vals[0];
	seg->coeffs[1] = m0;
	seg->coeffs[2] = - vals[0] - m0 + vals[1];
	seg->coeffs[3] = 0.0;
	seg->area = area;
	area = seg->integral(1.0);
}

Spline::~Spline()
{
	delete[] segments;
}

Spline &Spline::operator = (const Spline &orig)
{
	delete[] segments;
	n = orig.n;
	segments = new Segment[n];
	std::memcpy(segments, orig.segments, sizeof(Segment) * n);
	area = orig.area;
	return *this;
}

float Spline::eval(float x) const
{
	if (x <= 0.0)
		return segments[0].eval(0.0);
	if (x >= 1.0)
		return segments[n - 1].eval(1.0);

	float total;
	float rest = std::modf(x * n, &total);

	return segments[(unsigned int)total].eval(rest);
}

float Spline::deriv(float x) const
{
	if (x < 0.0 || x > 1.0)
		return 0.0;
	else if (x == 0.0)
		return segments[0].deriv(0.0);
	else if (x == 1.0)
		return segments[n - 1].deriv(1.0);

	float total;
	float rest = std::modf(x * n, &total);

	return segments[(unsigned int)total].deriv(rest);
}

float Spline::integral(float x) const
{
	if (x <= 0.0)
		return 0.0;
	if (x >= 1.0)
		return 1.0;

	if (area < 1.0e-9)
		return 0.0;

	float total;
	float rest = std::modf(x * n, &total);

	return segments[(unsigned int)total].integral(rest) / area;
}

} // namespace PhysicsTools
