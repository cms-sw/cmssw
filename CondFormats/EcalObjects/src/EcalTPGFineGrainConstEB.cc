#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainConstEB.h"


EcalTPGFineGrainConstEB::EcalTPGFineGrainConstEB()
  : ThresholdETLow_(0), ThresholdETHigh_(0), RatioLow_(0), RatioHigh_(0), LUT_(0)
{ }

EcalTPGFineGrainConstEB::~EcalTPGFineGrainConstEB()
{ }

void EcalTPGFineGrainConstEB::getValues(uint32_t & ThresholdETLow, 
					uint32_t & ThresholdETHigh,
					uint32_t & RatioLow,
					uint32_t & RatioHigh,
					uint32_t & LUT) const 
{
  ThresholdETLow = ThresholdETLow_ ;
  ThresholdETHigh = ThresholdETHigh_ ;
  RatioLow = RatioLow_ ;
  RatioHigh = RatioHigh_ ;
  LUT = LUT_ ;
}

void EcalTPGFineGrainConstEB::setValues(const uint32_t & ThresholdETLow, 
				       const uint32_t & ThresholdETHigh,
				       const uint32_t & RatioLow,
				       const uint32_t & RatioHigh,
				       const uint32_t & LUT)
{
  ThresholdETLow_ = ThresholdETLow ;
  ThresholdETHigh_ = ThresholdETHigh ;
  RatioLow_ = RatioLow ;
  RatioHigh_ = RatioHigh ;
  LUT_ = LUT ;
}
