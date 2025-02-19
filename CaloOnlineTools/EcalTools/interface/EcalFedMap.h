#ifndef ECALFEDMAP_h
#define ECALFEDMAP_h

#include <map>
#include <string>

class EcalFedMap{

 public: 

  EcalFedMap();
  ~EcalFedMap();
  int getFedFromSlice(std::string);
  std::string getSliceFromFed(int);

 private:
  // use:
  // #include <boost/bimap.hpp>
  // bimap< int, std::string > bm; 
  // when available

  std::map<int,  std::string> fedToSliceMap_;
  std::map<std::string, int>  sliceToFedMap_;
};
#endif
