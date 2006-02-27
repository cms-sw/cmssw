#ifndef LinearFitErrorsIn2Coord_H
#define LinearFitErrorsIn2Coord_H

#include <vector>
using namespace std;

  /** Straight line fit for data with errors on both coordinates
   *  source: Numerical Recipes
   */

class LinearFitErrorsIn2Coord {

public:

  /** Approached slope: 
   *  - rescale y and sigy by var(x)/var(y)
   *  - fit a straight line with weights derived from 
   *  the scaled sum sigx^2 + sigy^2
   */
  float slope(const vector<float> & x, const vector<float> & y, int ndat, 
	      const vector<float> & sigx, const vector<float> & sigy) const;

  /** Approached intercept computed with approached slope
   */
  float intercept(const vector<float> & x, const vector<float> & y, int ndat, 
		  const vector<float> & sigx, const vector<float> & sigy) const;

private:

  float variance(const vector<float> & x, int ndat) const;


};

#endif
