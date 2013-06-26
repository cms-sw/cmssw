#ifndef LinearFitErrorsIn2Coord_H
#define LinearFitErrorsIn2Coord_H

#include <vector>

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
  float slope(const std::vector<float> & x, const std::vector<float> & y, int ndat, 
	      const std::vector<float> & sigx, const std::vector<float> & sigy) const;

  /** Approached intercept computed with approached slope
   */
  float intercept(const std::vector<float> & x, const std::vector<float> & y, int ndat, 
		  const std::vector<float> & sigx, const std::vector<float> & sigy) const;

private:

  float variance(const std::vector<float> & x, int ndat) const;


};

#endif
