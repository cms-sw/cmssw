#ifndef LinearFit_H
#define LinearFit_H

#include <vector>

/** Straight line fit for data with errors on one coordinate
 */

class LinearFit {

public:

  /** x_i, y_i: measurements of y(x_i), sigy_i: error (sigma) on y_i
   *  slope, intercept: fitted parameters
   *  covss, covii, covsi: covariance matrix of fitted parameters, 
   *  s denoting slope, i denoting intercept
   */
  void fit(const std::vector<float> & x, const std::vector<float> & y, int ndat, 
	   const std::vector<float> & sigy, float& slope, float& intercept, 
	   float& covss, float& covii, float& covsi) const;


};

#endif
