/**
 * Author: Paolo Meridiani
 * Created: 14 Nov 2006
 * $Id: $
 **/

#include "CondFormats/EcalObjects/interface/EcalChannelStatusCode.h"

EcalChannelStatusCode::EcalChannelStatusCode() {
  status_ = 0;
}

EcalChannelStatusCode::EcalChannelStatusCode(const EcalChannelStatusCode & ratio) {
  status_ = ratio.status_;
}

EcalChannelStatusCode::~EcalChannelStatusCode() {
}

EcalChannelStatusCode& EcalChannelStatusCode::operator=(const EcalChannelStatusCode& rhs) {
  status_ = rhs.status_;
  return *this;
}
