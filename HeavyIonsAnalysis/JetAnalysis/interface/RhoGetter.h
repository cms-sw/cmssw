
#ifndef __RHOyyy__
#define __RHOyyy__

#include <vector>

static double rhoBins[11] = {-5,-4,-3,-2,-1,0,1,2,3,4,5};

double getRho(double eta, const std::vector<double> rhos){
  int j = 0;
  double deta = 99;
  for(unsigned int i = 0; i < rhos.size() ; ++i){
    double d = fabs(eta-rhoBins[i]);
    if(d < deta){
      deta = d;
      j = i;
    }
  }
  return rhos[j];

  /*
    for(unsigned int i = 0; i < rhos.size()-1 ; ++i){
    if(eta > rhoBins[i]) j = i+1;
    }
    double r = rhos[j-1]*fabs(eta - rhoBins[j]) + rhos[j]*fabs(eta - rhoBins[j-1]);
    r /= fabs(rhoBins[j]-rhoBins[j-1]);
    return r;
  */

}

#endif
