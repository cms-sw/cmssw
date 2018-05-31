#ifndef RecoParticleFlow_PFClusterProducer_ComputeClusterTime_h
#define RecoParticleFlow_PFClusterProducer_ComputeClusterTime_h

// user include files
#include <algorithm>
#include <cmath>
#include <vector>


// functions to select the hits to compute the time of a given cluster
// start with the only hits with timing information
// average among the hits contained in the chosen time interval

// N.B. time is corrected wrt vtx-calorimeter distance 
// with straight line and light speed hypothesis
// for charged tracks or heavy particles (longer track length or beta < 1) 
// need to correct the offset at analysis level



namespace hgcalsimclustertime {

  //time-interval based on that ~210ps wide and with the highest number of hits
  float fixSizeHighestDensity(std::vector<float>& t, float deltaT=0.210 /*time window in ns*/, float timeWidthBy=0.5){

    float tolerance = 0.05f;
    std::sort(t.begin(), t.end());

    int max_elements = 0;
    int start_el = 0;
    int end_el = 0;
    float timeW = 0.f;

    for(auto start = t.begin(); start != t.end(); ++start) {
      const auto startRef = *start;
      int c = count_if(start, t.end(), [&](float el) {
	  return el - startRef <= deltaT + tolerance;
	});
      if (c > max_elements) {
	max_elements = c;
	auto last_el = find_if_not(start, t.end(), [&](float el) {
	    return el - startRef <= deltaT + tolerance;
	  });
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
    int num = 0;
    int totSize = t.size();

    for(int ij=0; ij<=start_el; ++ij){
      if(t[ij] > (t[start_el] - HalfTimeDiff) ){
	for(int kl=ij; kl<totSize; ++kl){
	  if(t[kl] < (t[end_el] + HalfTimeDiff) ){
	    sum += t[kl];
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
}

#endif
