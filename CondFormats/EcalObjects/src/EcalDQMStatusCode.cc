/**
 **/

#include "CondFormats/EcalObjects/interface/EcalDQMStatusCode.h"

EcalDQMStatusCode::EcalDQMStatusCode() { status_ = 0; }

EcalDQMStatusCode::EcalDQMStatusCode(const EcalDQMStatusCode& ratio) { status_ = ratio.status_; }

EcalDQMStatusCode::~EcalDQMStatusCode() {}

EcalDQMStatusCode& EcalDQMStatusCode::operator=(const EcalDQMStatusCode& rhs) {
  status_ = rhs.status_;
  return *this;
}
