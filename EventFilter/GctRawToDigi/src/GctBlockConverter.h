
#ifndef GCTBLOCKCONVERTER_H
#define GCTBLOCKCONVERTER_H

#include <vector>
#include <map>
#include <memory>



class GctBlockConverter {
 public:

  GctBlockConverter();
  ~GctBlockConverter();
  
  // recognised block ID
  bool validBlock(unsigned id);

  // return block length in 32-bit words
  unsigned blockLength(unsigned id);
    
  // template function to convert a block into a collection of RCT/GCT objects
  template<typename T>
    void convertBlock(const unsigned char * d, unsigned id, std::vector<T>* coll);

 private:

  // block info
  std::map<unsigned, unsigned> blockLength_;  // size of a block

};

#endif
