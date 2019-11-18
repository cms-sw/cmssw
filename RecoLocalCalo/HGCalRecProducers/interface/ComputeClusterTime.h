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
    std::pair<float,float> fixSizeHighestDensity(std::vector<float>& time,
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

    
    //useful for future developments - baseline for 0PU
    /*
    //time-interval based on the smallest one containing a minimum fraction of hits
    // vector with time values of the hit; fraction between 0 and 1; how much furher enlarge the selected time window
    float highestDensityFraction(std::vector<float>& hitTimes, float fractionToKeep=0.68, float timeWidthBy=0.5){
    
    std::sort(hitTimes.begin(), hitTimes.end());
    int totSize = hitTimes.size();
    int num = 0.;
    float sum = 0.;
    float minTimeDiff = 999.;
    int startTimeBin = 0;

    int totToKeep = int(totSize*fractionToKeep);
    int maxStart = totSize - totToKeep;

    for(int ij=0; ij<maxStart; ++ij){
      float localDiff = fabs(hitTimes[ij] - hitTimes[int(ij+totToKeep)]);
      if(localDiff < minTimeDiff){
	minTimeDiff = localDiff;
	startTimeBin = ij;
      }
    }

    // further adjust time width around the chosen one based on the hits density
    // proved to improve the resolution: get as many hits as possible provided they are close in time

    int startBin = startTimeBin;
    int endBin = startBin+totToKeep;
    float HalfTimeDiff = std::abs(hitTimes[startBin] - hitTimes[endBin]) * timeWidthBy;

    for(int ij=0; ij<startBin; ++ij){
      if(hitTimes[ij] > (hitTimes[startBin] - HalfTimeDiff) ){
	for(int kl=ij; kl<totSize; ++kl){
	  if(hitTimes[kl] < (hitTimes[endBin] + HalfTimeDiff) ){
	    sum += hitTimes[kl];
	    ++num;
	  }
	  else  break;
	}
	break;
      }
    }

    if(num == 0) return -99.;
    return sum/num;
  }
  */
}  // namespace hgcalsimclustertime

#endif


