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

