#ifndef CondFormats_EcalObjects_EcalDAQStatusCode_H
#define CondFormats_EcalObjects_EcalDAQStatusCode_H
/**
 * Author: Paolo Meridiani
 * Created: 14 Nov 2006
 * $Id: EcalDAQStatusCode.h,v 1.1 2010/01/29 10:48:37 fra Exp $
 **/


#include <iostream>
#include <boost/cstdint.hpp>

class EcalDAQStatusCode {
  public:
    EcalDAQStatusCode();
    EcalDAQStatusCode(const EcalDAQStatusCode & codeStatus);
    EcalDAQStatusCode(const uint16_t& encodedStatus) : status_(encodedStatus) {};
    ~EcalDAQStatusCode();

    //get Methods to be defined according to the final definition

    void print(std::ostream& s) const { s << "status is: " << status_; }

    EcalDAQStatusCode& operator=(const EcalDAQStatusCode& rhs);
    uint16_t getStatusCode() const { return status_; }

  private:
    uint16_t status_;
};
#endif
