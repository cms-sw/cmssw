///this file contains additional dynamic_bitset methods


#include "EventFilter/CSCRawToDigi/src/bitset_append.h" 
#include <boost/dynamic_bitset.hpp>



namespace bitset_utilities {

  ///this method takes two bitsets bs1 and bs2 and returns result of bs2 appended to the end of bs1
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

  ///this method takes numberOfBits bits from unsigned short * array and returns them in the bitset obj.
  boost::dynamic_bitset<> ushortToBitset(const unsigned int numberOfBits,
					 unsigned short * buf)
  {
    boost::dynamic_bitset<> result(numberOfBits);
    for(unsigned i = 0; i < result.size(); ++i)
      {
        result[i] = (buf[i/16]>>(i%16))&&0x0001;
      }
    return result;
  }


  ///this method takes bitset obj and returns char * array 
  void bitsetToChar(const boost::dynamic_bitset<> & bs, unsigned char * result)
  {
    for(unsigned i = 0; i < bs.size(); ++i)
      {
        result[i/8] =  (bs[i+7]<<7)+
	  (bs[i+6]<<6)+
	  (bs[i+5]<<5)+
	  (bs[i+4]<<4)+ 
	  (bs[i+3]<<3)+
	  (bs[i+2]<<2)+
	  (bs[i+1]<<1)+
	  bs[i];
	i+=7;

      }
  }



}
