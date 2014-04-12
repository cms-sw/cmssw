#include "CondFormats/ESObjects/interface/ESGain.h"

ESGain::ESGain() 
{
  gain_=0.;
}

ESGain::ESGain(const float & gain) {
  gain_ = gain;
}

ESGain::~ESGain() {

}
