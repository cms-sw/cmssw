/**
 * Author: Paolo Meridiani
 * Created: 14 Nov 2006
 * $Id: EcalChannelStatusCode.cc,v 1.1 2006/11/16 18:19:45 meridian Exp $
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
