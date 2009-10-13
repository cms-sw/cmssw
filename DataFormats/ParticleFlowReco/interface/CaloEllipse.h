/*
 * CaloEllipse.h
 *
 *  Created on: 24-Mar-2009
 *      Author: jamie
 */

#ifndef CALOELLIPSE_H_
#define CALOELLIPSE_H_

#include <utility>
#include <vector>
#include <iostream>
#include <string>
#include <sstream>
namespace pftools {

typedef std::pair<double, double> Point;
typedef std::vector<Point> PointVector;
typedef PointVector::const_iterator PointCit;
typedef PointVector::iterator PointIt;

class CaloEllipse {
public:
	CaloEllipse();

	virtual ~CaloEllipse();

	void addPoint(double x, double y) {
		std::pair<double, double> point(x, y);
		dataPoints_.push_back(point);
	}

	void clearPoints() {
		dataPoints_.clear();
	}

	Point getPosition() const;

	Point getMajorMinorAxes(double sigma = 1.0) const;

	double getTheta() const;

	double getEccentricity() const;

	double cachedTheta_;
	double cachedMajor_;
	double cachedMinor_;
	double cachedEccentricity_;

	void makeCaches();

	void resetCaches();

	void reset();


private:
	std::vector<Point> dataPoints_;

};
	std::ostream& operator<<(std::ostream& s, const CaloEllipse& em);
}


#endif /* CALOELLIPSE_H_ */

#ifndef TOSTR_H
#define TOSTR_H
#include <sstream>
#include <iostream>
template <class T> std::string toString(const std::pair<T, T>& aT) {
	std::ostringstream oss;
	oss << "(" << aT.first << ", " << aT.second << ")";
	return oss.str();
}
#endif /*TOSTR_H */


