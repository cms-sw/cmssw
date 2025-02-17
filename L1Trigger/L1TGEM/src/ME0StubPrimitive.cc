#include "L1Trigger/L1TGEM/interface/ME0StubPrimitive.h"

//define class ME0StubPrimitive
ME0StubPrimitive::ME0StubPrimitive() : lc{0}, hc{0}, id{0}, strip{0}, partition{0} { update_quality(); }
ME0StubPrimitive::ME0StubPrimitive(int lc_, int hc_, int id_, int strip_, int partition_)
    : lc{lc_}, hc{hc_}, id{id_}, strip{strip_}, partition{partition_} {
  update_quality();
}
ME0StubPrimitive::ME0StubPrimitive(int lc_, int hc_, int id_, int strip_, int partition_, double bx_)
    : lc{lc_}, hc{hc_}, id{id_}, strip{strip_}, partition{partition_}, bx{bx_} {
  update_quality();
}
ME0StubPrimitive::ME0StubPrimitive(
    int lc_, int hc_, int id_, int strip_, int partition_, double bx_, std::vector<double>& centroid_)
    : lc{lc_}, hc{hc_}, id{id_}, strip{strip_}, partition{partition_}, bx{bx_}, centroid{centroid_} {
  update_quality();
}
void ME0StubPrimitive::reset() {
  lc = 0;
  hc = 0;
  id = 0;
  update_quality();
}
void ME0StubPrimitive::update_quality() {
  int idmask;
  if (lc) {
    if (ignore_bend) {
      idmask = 0xfe;
    } else {
      idmask = 0xff;
    }
    quality = (lc << 23) | (hc << 17) | ((id & idmask) << 12) | (strip << 4) | partition;
  } else {
    quality = 0;
  }
}
void ME0StubPrimitive::fit(int max_span) {
  if (id!=0) {
    std::vector<double> tmp;
    for (double cent : centroid) {
      tmp.push_back(cent-(max_span/2+1));
    }
    std::vector<double> x;
    std::vector<double> centroids;
    for (uint32_t i=0; i < tmp.size(); ++i) {
      if (tmp[i] != -1*(max_span/2+1)) {
        x.push_back(i-2.5);
        centroids.push_back(tmp[i]);
      }
    }
    std::vector<double> fit = llse_fit(x, centroids);
    bend_ang = fit[0];
    substrip = fit[1];
    mse      = fit[2];
  }
}
std::vector<double> ME0StubPrimitive::llse_fit(const std::vector<double>& x, const std::vector<double>& y) {
  double x_sum = 0;
  double y_sum = 0;
  for (double val : x) {
    x_sum += val;
  }
  for (double val : y) {
    y_sum += val;
  }
  int n = x.size();
  // linear regression
  double product = 0;
  double squares = 0;
  for (int i = 0; i < n; ++i) {
    product += (n * x[i] - x_sum) * (n * y[i] - y_sum);
    squares += (n * x[i] - x_sum) * (n * x[i] - x_sum);
  }

  double m = product / squares;
  double b = (y_sum - m * x_sum) / n;
  double sse = 0.0;
  for (int i = 0; i < n; ++i) {
    sse += (y[i] - m * x[i] - b) * (y[i] - m * x[i] - b);
  }

  std::vector<double> fit = {m, b, sse / n};
  return fit;
}
