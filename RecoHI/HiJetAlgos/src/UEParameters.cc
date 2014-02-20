
#include "RecoHI/HiJetAlgos/interface/UEParameters.h"

UEParameters::UEParameters(const std::vector<float> *v, int nn, int neta) : v_(v), nn_(nn), neta_(neta){
  parameters_ = new boost::const_multi_array_ref<float, 4>(&(*v)[0], boost::extents[neta][nreduced_particle_flow_id][nn][2]);
}

void UEParameters::get_fourier(double &re, double &im, size_t n, size_t eta, int type) const {
  re = 0;
  im = 0;

  if(type < 0){
    for (size_t i = 0; i < nreduced_particle_flow_id; i++) {
      re += (*parameters_)[eta][i][n][0];
      im += (*parameters_)[eta][i][n][1];
    }
  }else{
    re = (*parameters_)[eta][type][n][0];
    im = (*parameters_)[eta][type][n][1];
  }

}

double UEParameters::get_sum_pt(int eta, int type) const {
  double re;
  double im;
  
  get_fourier(re, im, 0, eta, type);
  
  // There is no imaginary part
  return re;
}

double UEParameters::get_vn(int n, int eta, int type) const {
  if (n < 0) {
    return NAN;
  }
  else if (n == 0) {
    return 1;
  }
  
  double re;
  double im;
  
  get_fourier(re, im, n, eta, type);
  
  return sqrt(re * re + im * im) / get_sum_pt(eta, type);
}

double UEParameters::get_psin(int n, int eta, int type) const {
  if (n <= 0) {
    return 0;
  }
  
  double re;
  double im;
  
  get_fourier(re, im, n, eta, type);
  
  return atan2(im, re) / n;
}

