#include "RecoLocalTracker/SiStripZeroSuppression/interface/IteratedMedianCMNSubtractor.h"

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include <cmath>



// this part should be moved to a "utility" package
// unit tests need to be added
namespace {

  // fastest median search to date 
  //  http://ndevilla.free.fr/median/median/index.html
  // code adapted from http://ndevilla.free.fr/median/median/src/quickselect.c
  /*
   *  This Quickselect routine is based on the algorithm described in
   *  "Numerical recipes in C", Second Edition,
   *  Cambridge University Press, 1992, Section 8.5, ISBN 0-521-43108-5
   *  This code by Nicolas Devillard - 1998. Public domain.
   */
  inline float quick_select(float arr[], int n) {
    int low, high ;
    int median;
    int middle, ll, hh;
    
    low = 0 ; high = n-1 ; median = (low + high) / 2;
    for (;;) {
      if (high <= low) /* One element only */
	return arr[median] ;
      
      if (high == low + 1) {  /* Two elements only */
	if (arr[low] > arr[high])
	  std::swap(arr[low], arr[high]) ;
	return arr[median] ;
      }
      
      /* Find median of low, middle and high items; swap into position low */
      middle = (low + high) / 2;
      if (arr[middle] > arr[high])    std::swap(arr[middle], arr[high]) ;
      if (arr[low] > arr[high])       std::swap(arr[low], arr[high]) ;
      if (arr[middle] > arr[low])     std::swap(arr[middle], arr[low]) ;
      
      /* Swap low item (now in position middle) into position (low+1) */
      std::swap(arr[middle], arr[low+1]) ;
      
      /* Nibble from each end towards middle, swapping items when stuck */
      ll = low + 1;
      hh = high;
      for (;;) {
        do ll++; while (arr[low] > arr[ll]) ;
        do hh--; while (arr[hh]  > arr[low]) ;
	
        if (hh < ll)
	  break;
	
        std::swap(arr[ll], arr[hh]) ;
      }
      
      /* Swap middle item (in position low) back into correct position */
      std::swap(arr[low], arr[hh]) ;
      
      /* Re-set active partition */
      if (hh <= median)
        low = ll;
      if (hh >= median)
        high = hh - 1;
    }
    return 0.f; // compiler
  }
  

  namespace wirth {

    typedef float elem_type ;
    
    
    /*---------------------------------------------------------------------------
      Function :   kth_smallest()
      In       :   array of elements, # of elements in the array, rank k
      Out      :   one element
      Job      :   find the kth smallest element in the array
      Notice   :   use the median() macro defined below to get the median. 
      
      Reference:
      
      Author: Wirth, Niklaus 
      Title: Algorithms + data structures = programs 
      Publisher: Englewood Cliffs: Prentice-Hall, 1976 
      Physical description: 366 p. 
      Series: Prentice-Hall Series in Automatic Computation 
      
      ---------------------------------------------------------------------------*/
    
    
    inline elem_type kth_smallest(elem_type a[], int n, int k) {
      int i,j,l,m ;
      elem_type x ;
      
      l=0 ; m=n-1 ;
      while (l<m) {
        x=a[k] ;
        i=l ;
        j=m ;
        do {
	  while (a[i]<x) i++ ;
	  while (x<a[j]) j-- ;
	  if (i<=j) {
	    std::swap(a[i],a[j]) ;
	    i++ ; j-- ;
	  }
        } while (i<=j) ;
        if (j<k) l=i ;
        if (k<i) m=j ;
      }
      return a[k] ;
    }
    

    inline elem_type median(elem_type a[], int n) { return kth_smallest(a,n,(((n)&1)?((n)/2):(((n)/2)-1))); }
  }
}


#include "boost/iterator/filter_iterator.hpp"

namespace {
  struct SelectElem {
    std::pair<float,float>  const * begin;
    bool const * ok;
    SelectElem() : begin(0), ok(0){}
    SelectElem(std::pair<float,float> const * isample, bool const * iok):
      begin(isample), ok(iok){}
    bool operator()(std::pair<float,float> const & elem) const {
      return ok[&elem-begin];
    }
  };
  
  
  typedef boost::filter_iterator<SelectElem, std::pair<float,float> const *> ElemIterator;
  inline float pairMedian(ElemIterator b, ElemIterator e) {
    float sample[128]; int size=0;
    for (;b!=e; ++b) sample[size++] = (*b).first;
    //    return  wirth::median(sample,size);
    return  quick_select(sample,size);
  }
}


void IteratedMedianCMNSubtractor::init(const edm::EventSetup& es){
  uint32_t n_cache_id = es.get<SiStripNoisesRcd>().cacheIdentifier();
  uint32_t q_cache_id = es.get<SiStripQualityRcd>().cacheIdentifier();
  
  if(n_cache_id != noise_cache_id) {
    es.get<SiStripNoisesRcd>().get( noiseHandle );
    noise_cache_id = n_cache_id;
  }
  if(q_cache_id != quality_cache_id) {
    es.get<SiStripQualityRcd>().get( qualityHandle );
    quality_cache_id = q_cache_id;
  }
}

void IteratedMedianCMNSubtractor::subtract(const uint32_t& detId,std::vector<int16_t>& digis){ subtract_(detId,digis);}
void IteratedMedianCMNSubtractor::subtract(const uint32_t& detId,std::vector<float>& digis){ subtract_(detId,digis);}

template<typename T>
inline
void IteratedMedianCMNSubtractor::
subtract_(const uint32_t& detId,std::vector<T>& digis){


  SiStripNoises::Range detNoiseRange = noiseHandle->getRange(detId);
  SiStripQuality::Range detQualityRange = qualityHandle->getRange(detId);

  typename std::vector<T>::iterator fs,ls;
  float offset = 0;  
  std::pair<float,float> subset[128];
  bool ok[128];
  int subsetSize=0;
  typedef std::pair<float,float> const * iterator;
  SelectElem selector(subset,ok);
  
  _vmedians.clear(); 

  for( uint16_t APV=0; APV< digis.size()/128; ++APV)
  {
    subsetSize=0;
    // fill subset vector with all good strips and their noises
    for (uint16_t istrip=APV*128; istrip<(APV+1)*128; ++istrip)
    {
      if ( !qualityHandle->IsStripBad(detQualityRange,istrip) )
      {
        std::pair<float,float> pin((float)digis[istrip], (float)noiseHandle->getNoiseFast(istrip,detNoiseRange));
        subset[subsetSize]= pin;
	ok[subsetSize++]=true;
      }
    }

    if (subsetSize == 0) continue;

    // std::cout << "subset size " << subsetSize << std::endl;

    ElemIterator begin(selector,subset,subset+subsetSize);
    ElemIterator end(selector,subset+subsetSize,subset+subsetSize);

    // caluate offset for all good strips (first iteration)
    offset = pairMedian(begin,end);

    // for second, third... iterations, remove strips over threshold
    // and recalculate offset on remaining strips
    int nokold=subsetSize;
    for ( int ii = 0; ii<iterations_-1; ++ii )
    {
      int nok=0;
      for (int j=0; j!=subsetSize;++j)
      {
	iterator si = subset+j;
	ok[j] =  si->first-offset < cut_to_avoid_signal_*si->second;
	++nok;
      }
      if (nok==nokold) break;   // std::cout << "converged at " << ii << std::endl;
      if (nok == 0 ) break;
      offset = pairMedian(begin,end);
      nokold=nok;
    }        

    _vmedians.push_back(std::pair<short,float>(APV,offset));

    // remove offset
    fs = digis.begin()+APV*128;
    ls = digis.begin()+(APV+1)*128;
    while (fs < ls) {
      *fs = static_cast<T>(*fs-offset);
      fs++;
    }

  }
  // std::cout << "IMCMNS end " <<  _vmedians.size() << std::endl;

}



