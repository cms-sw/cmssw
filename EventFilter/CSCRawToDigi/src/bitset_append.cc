#include "EventFilter/CSCRawToDigi/src/bitset_append.h" 
#include <boost/dynamic_bitset.hpp>

namespace bitset_utilities {
  boost::dynamic_bitset<> append(const boost::dynamic_bitset<> & bs1, 
				 const boost::dynamic_bitset<> & bs2)
  {
    boost::dynamic_bitset<> result(bs1.size()+bs2.size());
    unsigned size1 = bs1.size();
    for(unsigned i = 0; i < size1; ++i)
      {
	result[i] = bs1[i];
      }
    for(unsigned i = 0; i < bs2.size(); ++i)
      {
	result[size1+i] = bs2[i];
      }
    return result;
  }
}
