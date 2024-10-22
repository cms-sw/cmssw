/**
 * Author: Paolo Meridiani
 * Created: 14 Nov 2006
 * $Id: EcalTPGCrystalStatusCode.cc,v 1.1 2006/11/16 18:19:45 meridian Exp $
 **/

#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatusCode.h"

EcalTPGCrystalStatusCode::EcalTPGCrystalStatusCode() { status_ = 0; }

EcalTPGCrystalStatusCode::EcalTPGCrystalStatusCode(const EcalTPGCrystalStatusCode& ratio) { status_ = ratio.status_; }

EcalTPGCrystalStatusCode::~EcalTPGCrystalStatusCode() {}

EcalTPGCrystalStatusCode& EcalTPGCrystalStatusCode::operator=(const EcalTPGCrystalStatusCode& rhs) {
  status_ = rhs.status_;
  return *this;
}
