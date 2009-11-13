#include "DataFormats/Common/interface/MergeableCounter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace edm 
{

  bool MergeableCounter::mergeProduct(MergeableCounter const& a) 
  {
    if (a.value > 0 && value+a.value < a.value){
      edm::LogWarning("MergeableCounter|ProductsNotMergeable")
	<< "The merge would lead to an overflow of the counter" << std::endl;
      return false;
    }
    value += a.value;
    return true;
  }

}
