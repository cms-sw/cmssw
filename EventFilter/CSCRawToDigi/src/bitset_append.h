#ifndef BITSET_APPEND_H
#define BITSET_APPEND_H
#include <boost/dynamic_bitset.hpp>

namespace bitset_utilities {
   boost::dynamic_bitset<> append(const boost::dynamic_bitset<> & bs1, 
				  const boost::dynamic_bitset<> & bs2);
}
 
#endif
