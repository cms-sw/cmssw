/**
 * Author: Shahram Rahatlou, University of Rome & INFN
 * Created: 22 Feb 2006
 * $Id: EcalMGPAGainRatio.cc,v 1.3 2006/02/23 16:56:35 rahatlou Exp $
 **/
#include "CondFormats/EcalObjects/interface/EcalMGPAGainRatio.h"

EcalMGPAGainRatio::EcalMGPAGainRatio() {
  gain12Over6_ = 2.;
  gain6Over1_  = 6.;
}

EcalMGPAGainRatio::EcalMGPAGainRatio(const EcalMGPAGainRatio & ratio) {
  gain12Over6_ = ratio.gain12Over6_;
  gain6Over1_  = ratio.gain6Over1_;
}

EcalMGPAGainRatio::~EcalMGPAGainRatio() {

}

EcalMGPAGainRatio& EcalMGPAGainRatio::operator=(const EcalMGPAGainRatio& rhs) {
  gain12Over6_ = rhs.gain12Over6_;
  gain6Over1_ = rhs.gain6Over1_;
  return *this;
}
