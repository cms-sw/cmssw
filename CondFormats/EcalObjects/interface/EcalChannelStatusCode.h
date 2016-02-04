#ifndef CondFormats_EcalObjects_EcalChannelStatusCode_H
#define CondFormats_EcalObjects_EcalChannelStatusCode_H
/**
 * Author: Paolo Meridiani
 * Created: 14 Nov 2006
 * $Id: EcalChannelStatusCode.h,v 1.2 2008/02/18 10:49:28 ferriff Exp $
 **/


#include <iostream>
#include <boost/cstdint.hpp>

class EcalChannelStatusCode {
  public:
    EcalChannelStatusCode();
    EcalChannelStatusCode(const EcalChannelStatusCode & codeStatus);
    EcalChannelStatusCode(const uint16_t& encodedStatus) : status_(encodedStatus) {};
    ~EcalChannelStatusCode();

    //get Methods to be defined according to the final definition

    void print(std::ostream& s) const { s << "status is: " << status_; }

    EcalChannelStatusCode& operator=(const EcalChannelStatusCode& rhs);
    uint16_t getStatusCode() const { return status_; }

  private:
    uint16_t status_;
};
#endif
