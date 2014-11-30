#ifndef Common_MergeableHisto_h
#define Common_MergeableHisto_h

#include <vector>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace edm {
  
  template<class T>
    struct MergeableHisto {
      typedef T value_type;
      typedef typename std::vector<T> container_type;
      ~MergeableHisto() {}
      bool mergeProduct(MergeableHisto<T> const & a) {
	if(a.min != min || a.max != max || a.values.size() != values.size() ){
	  edm::LogWarning("MergeabloHisto|ProductsNotMergeable")
	    << "Trying to merge histograms with different binnings\n";
	  return false;
	}
	
	for(size_t ib=0; ib<values.size(); ++ib) {  values[ib]+=a.values[ib]; }
	return true;
      }
  
      value_type min, max;
      container_type values;
    };

  typedef MergeableHisto<float> MergeableHistoF;
  typedef MergeableHisto<int>   MergeableHistoI;
  typedef MergeableHisto<double> MergeableHistoD;

}

#endif
