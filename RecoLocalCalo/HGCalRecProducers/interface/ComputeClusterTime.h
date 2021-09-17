#ifndef RecoLocalCalo_HGCalRecProducers_ComputeClusterTime_h
#define RecoLocalCalo_HGCalRecProducers_ComputeClusterTime_h

// user include files
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <string>

// functions to select the hits to compute the time of a given cluster
// start with the only hits with timing information
// average among the hits contained in the chosen time interval
// weighted average wrt resolution or preferred function

// N.B. time is corrected wrt vtx-calorimeter distance
// with straight line and light speed hypothesis
// for charged tracks or heavy particles (longer track length or beta < 1)
// need to correct the offset at analysis level

namespace hgcalsimclustertime {

  class ComputeClusterTime {
  public:
    ComputeClusterTime(float Xmix, float Xmax, float Cterm, float Aterm);
    ComputeClusterTime();

    void setParameters(float Xmix, float Xmax, float Cterm, float Aterm);

    //time resolution parametrization
    float timeResolution(float xVal);

    float getTimeError(std::string type, float xVal);

    //time-interval based on that ~210ps wide and with the highest number of hits
    //apply weights if provided => weighted mean
    //return also error on the mean
    //only effective with a minimum number of hits with time (3 from TDR time)
    std::pair<float, float> fixSizeHighestDensity(std::vector<float>& time,
                                                  std::vector<float> weight = std::vector<float>(),
                                                  unsigned int minNhits = 3,
                                                  float deltaT = 0.210, /*time window in ns*/
                                                  float timeWidthBy = 0.5);

  private:
    float xMin_;
    float xMax_;
    float cTerm_;
    float aTerm_;
  };

}  // namespace hgcalsimclustertime

#endif
