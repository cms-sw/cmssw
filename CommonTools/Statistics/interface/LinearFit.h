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

// template version, no std (error alrady double...)
template<typename T> 
void linearFit( T const  * __restrict__ x, T const  * __restrict__ y, int ndat,
		T const  * __restrict__ sigy2,  
		T & slope, T & intercept,
		T & covss, T & covii, T & covsi) {
  T g1 = 0, g2 = 0;
  T s11 = 0, s12 = 0, s22 = 0;
  for (int i = 0; i != ndat; i++) {
    T sy2 = T(1)/sigy2[i];
    g1 += y[i] *sy2;
    g2 += x[i]*y[i] * sy2;
    s11 += sy2;
    s12 += x[i] * sy2;
    s22 += x[i]*x[i] * sy2;
  }
  
  T d = T(1)/(s11*s22 - s12*s12);
  intercept = (g1*s22 - g2*s12) * d;
  slope = (g2*s11 - g1*s12) * d;
  
  covii =  s22 * d;
  covss =  s11 * d;
  covsi = -s12 * d;
}


#endif
