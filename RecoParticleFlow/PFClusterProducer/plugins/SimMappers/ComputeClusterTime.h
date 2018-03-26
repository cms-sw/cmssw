// user include files
#include <TH1F.h>
#include <vector>


// functions to select the hits to compute the time of a given cluster
// start with the only hits with timing information
// average among the hits contained in the chosen time interval

// N.B. time is corrected wrt vtx-calorimeter distance 
// with straight line and light speed hypothesis
// for charged tracks or heavy particles (longer track length or beta < 1) 
// need to correct the offset at analysis level



//time-interval based on the smallest one containing a minimum fraction of hits
float highestDensityFraction(std::vector<float>& hitTimes, float fractionToKeep=0.68 /*fraction between 0 and 1*/, float timeWidthBy=0.5){

  std::sort(hitTimes.begin(), hitTimes.end());
  int totSize = hitTimes.size();

  int num = 0.;
  float sum = 0.;

  float minTimeDiff = 999.;
  int startTimeBin = 0;
  
  int startBin = 0;
  int endBin = totSize;
  
  for(int ij=0; ij<int(totSize*(1.-fractionToKeep)); ++ij){
    float localDiff = fabs(hitTimes.at(ij) - hitTimes.at(int(ij+totSize*fractionToKeep)));
    if(localDiff < minTimeDiff){
      minTimeDiff = localDiff;
      startTimeBin = ij;
    }
  }

  // further adjust time width around the chosen one based on the hits density
  // proved to improve the resolution: get as many hits as possible provided they are close in time
  startBin = startTimeBin;
  endBin = int(startBin+totSize*fractionToKeep);
  float HalfTimeDiff = std::abs(hitTimes.at(startBin) - hitTimes.at(endBin)) * timeWidthBy;

  for(int ij=0; ij<startBin; ++ij){  
    if(hitTimes.at(ij) > (hitTimes.at(startBin) - HalfTimeDiff) ){
      for(int kl=ij; kl<totSize; ++kl){   
	if(hitTimes.at(kl) < (hitTimes.at(endBin) + HalfTimeDiff) ){	 	  
	  sum += hitTimes.at(kl); 
	  ++num;
	}
	else  break;                                                                                                                                                                                                          }
      break;
    }                                                                                                                                                                                                          
  }
  
  if(num == 0) return -99.;  
  return sum/num;
}



//time-interval based on that ~210ps wide and with the highest number of hits
float highestDensityTimeDiff(std::vector<float>& hitTimes, float deltaToKeep=0.210 /*time window in ns*/, float timeWidthBy=0.5){

  std::sort(hitTimes.begin(), hitTimes.end());
  int totSize = hitTimes.size();

  float minEntries = 0;
  float deltaEpsilon = 0.05; //TDCfor ToA has bins of 24.5ps

  int startBin = 0;
  int endBin = 0;

  float timeDiff = 0.;

  int num = 0.;
  float sum = 0.;

  for(int iS=0; iS<totSize; ++iS){
    for(int jS=iS; jS<totSize; ++jS){
      
      float wTime = hitTimes.at(jS) - hitTimes.at(iS);
      int wEntries = (jS - iS) + 1;
      
      if( std::abs(wTime - deltaToKeep) <= deltaEpsilon || (wTime < deltaToKeep && wEntries > minEntries) ){
	if(wEntries > minEntries){
	  if(std::abs(wTime - deltaToKeep) <= deltaEpsilon) deltaEpsilon = std::abs(wTime - deltaToKeep);
	  startBin = iS;
	  endBin = jS;
	  timeDiff = wTime;
	  minEntries = wEntries;
	}
      }
    }
  }

  // further adjust time width around the chosen one based on the hits density
  // proved to improve the resolution: get as many hits as possible provided they are close in time
  float HalfTimeDiff = timeDiff * timeWidthBy;

  for(int ij=0; ij<=startBin; ++ij){     
    if(hitTimes.at(ij) > (hitTimes.at(startBin) - HalfTimeDiff) ){  
      for(int kl=ij; kl<totSize; ++kl){
        if(hitTimes.at(kl) < (hitTimes.at(endBin) + HalfTimeDiff) ){
          sum += hitTimes.at(kl);
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
