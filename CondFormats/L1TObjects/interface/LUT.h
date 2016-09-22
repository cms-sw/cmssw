///
/// \class l1t::LUT
///
/// Description: A class implimentating a look up table
///
/// Implementation:
///    Internally stores data in vector filled with 32 bit ints
///    address and output is masked to specifed number of bits
///    vector only allocates as necessary, eg we may have a 32 bit address but we obviously dont want to allocate a 4gb vector 
///
/// Error handling: currently waiting guidance on how to deal with this
/// Exceptions are currently forbiden, emulator fails should not impact on the rest of the software chain. As such, everything silently fails gracefully
/// \author: Sam Harper - RAL, Jim Brooke - Bristol
///

//

#ifndef CondFormats_L1TObjects_LUT_h
#define CondFormats_L1TObjects_LUT_h

#include <iostream>
#include <vector>
#include <limits>

#include "CondFormats/Serialization/interface/Serializable.h"

namespace l1t{

  class LUT{
    public:
    enum ReadCodes {
      SUCCESS=0,NO_ENTRIES=1,DUP_ENTRIES=2,MISS_ENTRIES=3,MAX_ADDRESS_OUTOFRANGE=4,NO_HEADER=5
    };
    
    LUT():data_(){}
      
    explicit LUT(std::istream& stream) :
      data_()
    {
      read(stream);
    }
	
	
    ~LUT(){}
    
    int data(unsigned int address)const{return (address&addressMask_)<data_.size() ? dataMask_ & data_[addressMask_&address] : 0;}
    int read(std::istream& stream);
    void write(std::ostream& stream)const;

    unsigned int nrBitsAddress()const{return nrBitsAddress_;}
    unsigned int nrBitsData()const{return nrBitsData_;}
    //following the convention of vector::size()
    unsigned int maxSize()const{return addressMask_==std::numeric_limits<unsigned int>::max() ? addressMask_ : addressMask_+1;}
    bool empty()const{return data_.empty();}
   
  private:
    
    int readHeader_(std::istream&);

    unsigned int nrBitsAddress_; //technically redundant with addressMask
    unsigned int nrBitsData_;//technically redundant with dataMask
    unsigned int addressMask_;
    unsigned int dataMask_;
   
    std::vector<int> data_;
    COND_SERIALIZABLE;
  };
    
}

#endif
