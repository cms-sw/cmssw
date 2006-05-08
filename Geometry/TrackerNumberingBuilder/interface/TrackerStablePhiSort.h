#ifndef TrackerStablePhiSort_H
#define TrackerStablePhiSort_H
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <utility>
#include <vector>
#include <algorithm>

//#define DEBUG
namespace {
  template<class T, class Scalar>
  struct LessPair {
    typedef std::pair<T,Scalar> SortPair;
    bool operator()( const SortPair& a, const SortPair& b) {
      return a.second < b.second;
    }
  };
}


template<class RandomAccessIterator, class Extractor>
void TrackerStablePhiSort(RandomAccessIterator begin,
			    RandomAccessIterator end,
			    const Extractor& extr) {

  typedef typename Extractor::result_type        Scalar;
  typedef std::pair<RandomAccessIterator,Scalar> SortPair;

  std::vector<SortPair> tmpvec; 
  tmpvec.reserve(end-begin);

  std::vector<SortPair> tmpcop; 
  tmpcop.reserve(end-begin);

  std::vector<SortPair> copy1; 
  copy1.reserve(end-begin);

  std::vector<SortPair> copy2; 
  copy2.reserve(end-begin);
  
  // tmpvec holds iterators - does not copy the real objects
  for (RandomAccessIterator i=begin; i!=end; i++) {
    tmpvec.push_back(SortPair(i,extr(*i)));
  }
  
  std::sort(tmpvec.begin(), tmpvec.end(),
	    LessPair<RandomAccessIterator,Scalar>());    

  //stability check
  double pi = 3.141592653592;
  unsigned int vecSize = tmpvec.size();
  bool check = false;
  if(vecSize>1){    
    for(unsigned int i = 0;i <vecSize; i++){
      double res = tmpvec[i].second <= pi? tmpvec[i].second/3. : fabs(2*pi-tmpvec[i].second)/3.;

      LogDebug("StableSort")<<"Component sorted # "<<i;
      LogDebug("StableSort")<<" Phi = "<<tmpvec[i].second<<" resolution = "<<res;

      double dist = std::max(res,0.001);
      double dist_tec = i>0 ? std::min(fabs(tmpvec[i].second-tmpvec[i-1].second),1.):
	std::min(fabs(tmpvec[i+1].second-tmpvec[i].second),1.);
      if(dist==0.001){

	LogDebug("StableSort")<<"Object close to 0";
      
	copy1.insert(copy1.begin(),tmpvec[i]);
	tmpcop.insert(tmpcop.begin(),tmpvec[i]);
	  if(check){
	    edm::LogError("StableSort")<<"Two modules are close to 0";
	    abort();
	  }
	check= true;
      }else{
	copy1.push_back(tmpvec[i]);
	tmpcop.push_back(tmpvec[i]);
      }


      if(dist_tec==1.){
	bool check1 = false;
	tmpcop.clear();
	for(unsigned int jj=i+1;jj<vecSize;jj++){	  
	  copy1.push_back(SortPair(tmpvec[jj].first,(2*pi-tmpvec[jj].second)));
	}
	std::sort(copy1.begin(), copy1.end(),LessPair<RandomAccessIterator,Scalar>());

	for(unsigned int ii = 0;ii <vecSize; ii++){
	  double res = fabs(copy1[ii].second);

	  LogDebug("StableSort")<<"Component sorted again # "<<ii;
	  LogDebug("StableSort")<<" Phi = "<<tmpvec[i].second<<" resolution = "<<res;
      
	  double dist = std::max(res,0.001);
	  if(dist==0.001){

	    LogDebug("StableSort")<<"Object close to 0 again";
      
	    copy2.insert(copy2.begin(),copy1[ii]);
	    tmpcop.insert(tmpcop.begin(),copy1[ii]);
	    if(check1){
	      edm::LogError("StableSort")<<"Two modules are close to 0";
	      abort();
	    }
	    check1= true;
	  }else{
	    copy2.push_back(copy1[ii]);
	    tmpcop.push_back(copy1[ii]);
	  }
	}
	break;
      }
    }
  }

  // overwrite the input range with the sorted values
  // copy of input container not necessary, but tricky to avoid
  std::vector<typename std::iterator_traits<RandomAccessIterator>::value_type> tmpcopy(begin,end);
  if(vecSize=1){
    for (unsigned int i=0; i<tmpvec.size(); i++) {
      *(begin+i) = tmpcopy[tmpvec[i].first - begin];
    }    
  }
  for (unsigned int i=0; i<tmpcop.size(); i++) {
    *(begin+i) = tmpcopy[tmpcop[i].first - begin];
  }
};

#endif
