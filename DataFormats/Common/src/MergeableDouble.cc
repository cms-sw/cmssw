#include "DataFormats/Common/interface/MergeableDouble.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace edm 
{

  bool MergeableDouble::mergeProduct(MergeableDouble const& a) 
  {
    if (a.value > 0 && value+a.value < a.value){
      edm::LogWarning("MergeableDouble|ProductsNotMergeable")
	<< "The merge would lead to an overflow of the counter" << std::endl;
      return false;
    }
    value += a.value;
    return true;
  }

}
