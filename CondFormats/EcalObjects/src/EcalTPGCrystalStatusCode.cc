/**
 * Author: Paolo Meridiani
 * Created: 14 Nov 2006
 * $Id: EcalTPGCrystalStatusCode.cc,v 1.1 2008/12/03 15:09:24 fra Exp $
 **/

#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatusCode.h"

EcalTPGCrystalStatusCode::EcalTPGCrystalStatusCode() {
  status_ = 0;
}

EcalTPGCrystalStatusCode::EcalTPGCrystalStatusCode(const EcalTPGCrystalStatusCode & ratio) {
  status_ = ratio.status_;
}

EcalTPGCrystalStatusCode::~EcalTPGCrystalStatusCode() {
}

EcalTPGCrystalStatusCode& EcalTPGCrystalStatusCode::operator=(const EcalTPGCrystalStatusCode& rhs) {
  status_ = rhs.status_;
  return *this;
}
