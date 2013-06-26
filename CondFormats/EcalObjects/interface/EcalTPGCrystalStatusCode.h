#ifndef CondFormats_EcalObjects_EcalTPGCrystalStatusCode_H
#define CondFormats_EcalObjects_EcalTPGCrystalStatusCode_H
/**
 * Author: FC
 * Created: 3 dec 2008
 * 
 **/


#include <iostream>
#include <boost/cstdint.hpp>

class EcalTPGCrystalStatusCode {
  public:
    EcalTPGCrystalStatusCode();
    EcalTPGCrystalStatusCode(const EcalTPGCrystalStatusCode & codeStatus);
    EcalTPGCrystalStatusCode(const uint16_t& encodedStatus) : status_(encodedStatus) {};
    ~EcalTPGCrystalStatusCode();

    //get Methods to be defined according to the final definition

    void print(std::ostream& s) const { s << "status is: " << status_; }

    EcalTPGCrystalStatusCode& operator=(const EcalTPGCrystalStatusCode& rhs);
    uint16_t getStatusCode() const { return status_; }
    
    // for testing the L1 trigger emulator
    void setStatusCode(const uint16_t  & val) { status_=val; }
    
  private:
    uint16_t status_;
};
#endif
