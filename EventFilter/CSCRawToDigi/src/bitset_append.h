#ifndef BITSET_APPEND_H
#define BITSET_APPEND_H
#include <boost/dynamic_bitset.hpp>

namespace bitset_utilities {
   boost::dynamic_bitset<> append(const boost::dynamic_bitset<> & bs1, 
				  const boost::dynamic_bitset<> & bs2);

   boost::dynamic_bitset<> ushortToBitset(const unsigned int numberOfBits,
					  unsigned short * buf);
   void bitsetToChar(const boost::dynamic_bitset<> & bs, unsigned char * result);

   void printWords(const boost::dynamic_bitset<> & bs);


}
 
#endif
