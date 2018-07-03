/*
 * CaloEllipse.cc
 *
 *  Created on: 24-Mar-2009
 *      Author: jamie
 */

#include "DataFormats/ParticleFlowReco/interface/CaloEllipse.h"

#include <cmath>


using namespace pftools;

CaloEllipse::CaloEllipse() {


}

CaloEllipse::~CaloEllipse() {

}
double CaloEllipse::getTheta() const {
	if (dataPoints_.size() < 2) {
		return 0.0;
	}

	PointVector meanAdj;

	Point mean = getPosition();

	double sum_dxdx(0.0);
	double sum_dydy(0.0);
	double sum_dxdy(0.0);
	PointCit it = dataPoints_.begin();
	for (; it != dataPoints_.end(); ++it) {
		const Point& p = *it;
		Point q(p.first - mean.first, p.second - mean.second);
		meanAdj.push_back(q);
		sum_dxdx += q.first * q.first;
		sum_dydy += q.second * q.second;
		sum_dxdy += q.first * q.second;
	}
	double theta = 0.5 * atan(2.0 * sum_dxdy / (sum_dydy - sum_dxdx));
	return theta;
}

Point CaloEllipse::getMajorMinorAxes(double sigma) const {

	if (dataPoints_.size() < 2) {
		return Point(0, 0);
	}

	Point mean = getPosition();

	PointCit it = dataPoints_.begin();
	double sum_xx(0.0);
	double sum_yy(0.0);
	double theta = getTheta();

	for (; it != dataPoints_.end(); ++it) {
		const Point& p = *it;
		double X = cos(theta) * (p.first - mean.first) - sin(theta) * (p.second
				- mean.second);
		double Y = sin(theta) * (p.first - mean.first) + cos(theta) * (p.second
				- mean.second);
		sum_xx += X * X;
		sum_yy += Y * Y;
	}

	double a = sigma * sqrt(sum_xx / dataPoints_.size());
	double b = sigma * sqrt(sum_yy / dataPoints_.size());

	double major, minor;

	if(a > b) {
		major = a;
		minor = b;
	} else {
		major = b;
		minor = a;
	}

	return Point(major, minor);
}

double CaloEllipse::getEccentricity() const {
	Point p = getMajorMinorAxes();
	double a = p.first;
	double b = p.second;
	if(a == 0)
		return 0;
	double ecc = sqrt((a * a - b * b) / (a * a));
	return ecc;
}

Point CaloEllipse::getPosition() const {

	if (dataPoints_.empty()) {
		return Point(0, 0);
	}

	double x_tot(0.0);
	double y_tot(0.0);
	PointCit it = dataPoints_.begin();
	for (; it != dataPoints_.end(); ++it) {
		const Point& p = *it;
		x_tot += p.first;
		y_tot += p.second;

	}

	return Point(x_tot / dataPoints_.size(), y_tot / dataPoints_.size());
}

void CaloEllipse::resetCaches() {
	cachedTheta_ = 0.0;
	cachedMinor_ = 0.0;
	cachedMajor_ = 0.0;

}

void CaloEllipse::makeCaches() {
	cachedTheta_ = getTheta();
	Point axes = getMajorMinorAxes();
	cachedMajor_ = axes.first;
	cachedMinor_ = axes.second;
}

void CaloEllipse::reset() {
	dataPoints_.clear();
	resetCaches();
}

std::ostream& pftools::operator<<(std::ostream& s, const pftools::CaloEllipse& em) {
	s << "CaloEllipse at position = " << toString(em.getPosition()) << ", theta = "
			<< em.getTheta() << ", major/minor axes = " << toString(
			em.getMajorMinorAxes()) << ", eccentricity = " << em.getEccentricity() << "\n";

	return s;
}

