#include "RecoLocalCalo/HGCalRecProducers/interface/ComputeClusterTime.h"

// functions to select the hits to compute the time of a given cluster
// start with the only hits with timing information
// average among the hits contained in the chosen time interval

// N.B. time is corrected wrt vtx-calorimeter distance
// with straight line and light speed hypothesis
// for charged tracks or heavy particles (longer track length or beta < 1)
// need to correct the offset at analysis level

using namespace hgcalsimclustertime;

std::vector<size_t> decrease_sorted_indices(const std::vector<float>& v) {
  // initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indices based on comparing values in v (decreasing order)
  std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
  return idx;
};

ComputeClusterTime::ComputeClusterTime(float Xmin, float Xmax, float Cterm, float Aterm)
    : _Xmin(Xmin), _Xmax(Xmax), _Cterm(Cterm), _Aterm(Aterm) {
  if (_Xmin < 0)
    _Xmin = 0.1;
};

ComputeClusterTime::ComputeClusterTime() : _Xmin(1.), _Xmax(5.), _Cterm(0), _Aterm(0){};

void ComputeClusterTime::setParameters(float Xmin, float Xmax, float Cterm, float Aterm) {
  _Xmin = (Xmin > 0) ? Xmin : 0.1;
  _Xmax = Xmax;
  _Cterm = Cterm;
  _Aterm = Aterm;
  return;
}

//time resolution parametrization
float ComputeClusterTime::timeResolution(float x) {
  float funcVal = pow(_Aterm / x, 2) + pow(_Cterm, 2);
  return sqrt(funcVal);
}

float ComputeClusterTime::getTimeError(std::string type, float xVal) {
  if (type == "recHit") {
    //xVal is S/N
    //time is in ns units
    if (xVal < _Xmin)
      return timeResolution(_Xmin);
    else if (xVal > _Xmax)
      return _Cterm;
    else
      return timeResolution(xVal);

    return -1;
  }
  return -1;
}

//time-interval based on that ~210ps wide and with the highest number of hits
//extension valid in high PU of taking smallest interval with (order of)68% of hits
std::pair<float, float> ComputeClusterTime::fixSizeHighestDensity(std::vector<float>& time,
                                                                  std::vector<float> weight,
								  unsigned int minNhits,
                                                                  float deltaT,
                                                                  float timeWidthBy) {

  if(time.size() < minNhits)
    return std::pair<float, float>(-99., -1.);

  if (weight.empty())
    weight.resize(time.size(), 1.);

  std::vector<float> t(time.size(), 0.);
  std::vector<float> w(time.size(), 0.);
  std::vector<size_t> sortedIndex = decrease_sorted_indices(time);
  for (std::size_t i = 0; i < sortedIndex.size(); ++i) {
    t[i] = time[sortedIndex[i]];
    w[i] = weight[sortedIndex[i]];
  }

  int max_elements = 0;
  int start_el = 0;
  int end_el = 0;
  float timeW = 0.f;
  float tolerance = 0.05f;

  for (auto start = t.begin(); start != t.end(); ++start) {
    const auto startRef = *start;
    int c = count_if(start, t.end(), [&](float el) { return el - startRef <= deltaT + tolerance; });
    if (c > max_elements) {
      max_elements = c;
      auto last_el = find_if_not(start, t.end(), [&](float el) { return el - startRef <= deltaT + tolerance; });
      auto val = *(--last_el);
      if (std::abs(deltaT - (val - startRef)) < tolerance) {
        tolerance = std::abs(deltaT - (val - startRef));
      }
      start_el = distance(t.begin(), start);
      end_el = distance(t.begin(), last_el);
      timeW = val - startRef;
    }
  }

  // further adjust time width around the chosen one based on the hits density
  // proved to improve the resolution: get as many hits as possible provided they are close in time
  float HalfTimeDiff = timeW * timeWidthBy;
  float sum = 0.;
  float num = 0;
  int totSize = t.size();

  for (int ij = 0; ij <= start_el; ++ij) {
    if (t[ij] > (t[start_el] - HalfTimeDiff)) {
      for (int kl = ij; kl < totSize; ++kl) {
        if (t[kl] < (t[end_el] + HalfTimeDiff)) {
          sum += t[kl] * w[kl];
          num += w[kl];
        } else
          break;
      }
      break;
    }
  }

  if (num == 0) {
    return std::pair<float, float>(-99., -1.);
  }
  return std::pair<float, float>(sum / num, 1. / sqrt(num));
}
