#include "CondFormats/EcalObjects/interface/EcalTPGWeights.h"


EcalTPGWeights::EcalTPGWeights()
  : w0_(0), w1_(0), w2_(0), w3_(0), w4_(0)
{ }

EcalTPGWeights::~EcalTPGWeights()
{ }

void EcalTPGWeights::getValues(uint32_t & w0,
			       uint32_t & w1,
			       uint32_t & w2,
			       uint32_t & w3,
			       uint32_t & w4) const
{
  w0 = w0_ ;
  w1 = w1_ ;
  w2 = w2_ ;
  w3 = w3_ ;
  w4 = w4_ ;
}

void EcalTPGWeights::setValues(const uint32_t & w0,
			       const uint32_t & w1,
			       const uint32_t & w2,
			       const uint32_t & w3,
			       const uint32_t & w4)
{
  w0_ = w0 ;
  w1_ = w1 ;
  w2_ = w2 ;
  w3_ = w3 ;
  w4_ = w4 ;
}
