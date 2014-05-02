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

#ifndef L1Trigger_L1TCalorimeter_LUT_h
#define L1Trigger_L1TCalorimeter_LUT_h

#include <iostream>
#include <vector>
#include <limits>

namespace l1t{

  class LUT{
    public:
    
    // number of values stored is 2^nrBitsAddress
    // can store values between 0 and 2^(nrBitsData)-1
    LUT(unsigned int iNrBitsAddress=8, unsigned int iNrBitsData=8):  // number of bits address, and data 
      nrBitsAddress_(iNrBitsAddress),nrBitsData_(iNrBitsData),
      addressMask_((0x1<<nrBitsAddress_)-1),
      dataMask_((0x1<<nrBitsData_)-1),
      data_()
    {
      //impliment error checking for >32 bits for address or data 
    }
    
    LUT(unsigned int iNrBitsAddress, unsigned int iNrBitsData,std::istream& stream) :
      nrBitsAddress_(iNrBitsAddress),nrBitsData_(iNrBitsData),
      addressMask_((0x1<<nrBitsAddress_)-1),
      dataMask_((0x1<<nrBitsData_)-1),
      data_()
    {
      read(stream); 
    }
	
    ~LUT(){}
    
    int data(unsigned int address)const{return (address&addressMask_)<data_.size() ? dataMask_ & data_[addressMask_&address] : 0;}
    bool read(std::istream& stream);
    void write(std::ostream& stream)const;

    unsigned int nrBitsAddress()const{return nrBitsAddress_;}
    unsigned int nrBitsData()const{return nrBitsData_;}
    //following the convention of vector::size()
    unsigned int maxSize()const{return addressMask_==std::numeric_limits<unsigned int>::max() ? addressMask_ : addressMask_+1;}
   
  private:
    
    const unsigned int nrBitsAddress_; //technically redundant with addressMask
    const unsigned int nrBitsData_;//technically redundant with dataMask
    const unsigned int addressMask_;
    const unsigned int dataMask_;
    
    std::vector<int> data_;
  };
    
}

#endif
