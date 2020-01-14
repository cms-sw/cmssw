#ifndef RecoLocalCalo_HGCalRecProducers_ComputeClusterTime_h
#define RecoLocalCalo_HGCalRecProducers_ComputeClusterTime_h

// user include files
#include <algorithm>
#include <cmath>
#include <vector>
#include "TF1.h"

// functions to select the hits to compute the time of a given cluster
// start with the only hits with timing information
// average among the hits contained in the chosen time interval
// weighted average wrt resolution or preferred function

// N.B. time is corrected wrt vtx-calorimeter distance
// with straight line and light speed hypothesis
// for charged tracks or heavy particles (longer track length or beta < 1)
// need to correct the offset at analysis level

namespace hgcalsimclustertime {

  /*
  std::vector<size_t> decrease_sorted_indices(const std::vector<float>& v){
    // initialize original index locations                                       
    std::vector<size_t> idx(v.size());
    for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;
    // sort indices based on comparing values in v (decreasing order)            
    std::sort(idx.begin(), idx.end(),
	      [&v](size_t i1, size_t i2) {return v[i1] < v[i2];} );
    return idx;
  };
  */

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
    std::pair<float, float> fixSizeHighestDensity(std::vector<float>& time,
                                                  std::vector<float> weight = std::vector<float>(),
                                                  float deltaT = 0.210, /*time window in ns*/
                                                  float timeWidthBy = 0.5);

    //same as before but provide weights
    //configure the reweight through type
    /* std::pair<float,float> fixSizeHighestDensityEnergyResWeig(std::vector<float>& t, */
    /* 							      std::vector<float>& w, */
    /* 							      std::string& type, */
    /* 							      float deltaT = 0.210, //time window in ns */
    /* 							      float timeWidthBy = 0.5) {}; */

  private:
    float _Xmin;
    float _Xmax;
    float _Cterm;
    float _Aterm;
  };

}  // namespace hgcalsimclustertime

#endif
