/** \file
 *
 * $Date: 2007/04/24 13:50:13 $
 * $Revision: 1.2 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 */

/* This Class Header */
#include "RecoLocalMuon/DTSegment/src/DTLinearFit.h"

/* Collaborating Class Header */

/* C++ Headers */
using namespace std;

/* ====================================================================== */

/// Constructor
DTLinearFit::DTLinearFit() {

}

/// Destructor
DTLinearFit::~DTLinearFit() {
}

/* Operations */ 
void DTLinearFit::fit(const vector<float> & x,
                      const vector<float> & y, 
                      int ndat,
                      const vector<float> & sigy, 
                      float& slope,
                      float& intercept, 
                      float& covss,
                      float& covii,
                      float& covsi) const
{

  float g1 = 0, g2 = 0;
  float s11 = 0, s12 = 0, s22 = 0;
  for (int i = 0; i != ndat; i++) {
    float sy2 = sigy[i] * sigy[i];
    g1 += y[i] / sy2;
    g2 += x[i]*y[i] / sy2;
    s11 += 1. / sy2;
    s12 += x[i] / sy2;
    s22 += x[i]*x[i] / sy2;
  }

  float d = s11*s22 - s12*s12;
  intercept = (g1*s22 - g2*s12) / d;
  slope = (g2*s11 - g1*s12) / d;

  covii =  s22 / d;
  covss =  s11 / d;
  covsi = -s12 / d;

}
