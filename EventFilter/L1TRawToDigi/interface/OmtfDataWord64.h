#ifndef EventFilter_L1TRawToDigi_Omtf_DataWord64_H
#define EventFilter_L1TRawToDigi_Omtf_DataWord64_H 

#include<cstdint>
#include<iostream>
#include<vector>
#include<map>

namespace omtf {

typedef uint64_t Word64;

typedef std::map< std::pair<unsigned int, unsigned int>, std::vector<Word64> > FedAmcRawsMap;

namespace DataWord64 {
  enum Type { csc=0xC, dt= 0xD, rpc=0xE, omtf=0xF };
  template <typename T> Type type(const T&);
  template<> inline Type type<Word64>(const Word64 & data) { return static_cast<Type> (data>>60); }
  template<> inline Type type<unsigned int>(const unsigned int & intType) { return static_cast<Type> (intType); }
  std::ostream & operator<< (std::ostream &out, const Type &o);
};

} //namespace Omtf
#endif
