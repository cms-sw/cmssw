#include "CondFormats/ESObjects/interface/ESEEIntercalibConstants.h"

ESEEIntercalibConstants::ESEEIntercalibConstants() 
{
  gammaLow0_  = 0.;
  gammaHigh0_ = 0.;
  alphaLow0_  = 0.;
  alphaHigh0_ = 0.;

  gammaLow1_  = 0.;
  gammaHigh1_ = 0.;
  alphaLow1_  = 0.;
  alphaHigh1_ = 0.;

  gammaLow2_  = 0.;
  gammaHigh2_ = 0.;
  alphaLow2_  = 0.;
  alphaHigh2_ = 0.;

  gammaLow3_  = 0.;
  gammaHigh3_ = 0.;
  alphaLow3_  = 0.;
  alphaHigh3_ = 0.;
}

ESEEIntercalibConstants::ESEEIntercalibConstants(
  const float & gammaLow0, const float & alphaLow0, const float & gammaHigh0, const float & alphaHigh0, 
  const float & gammaLow1, const float & alphaLow1, const float & gammaHigh1, const float & alphaHigh1, 
  const float & gammaLow2, const float & alphaLow2, const float & gammaHigh2, const float & alphaHigh2, 
  const float & gammaLow3, const float & alphaLow3, const float & gammaHigh3, const float & alphaHigh3 
  ) 
{
  gammaLow0_  = gammaLow0;
  gammaHigh0_ = gammaHigh0;
  alphaLow0_  = alphaLow0;
  alphaHigh0_ = alphaHigh0;

  gammaLow1_  = gammaLow1;
  gammaHigh1_ = gammaHigh1;
  alphaLow1_  = alphaLow1;
  alphaHigh1_ = alphaHigh1;

  gammaLow2_  = gammaLow2;
  gammaHigh2_ = gammaHigh2;
  alphaLow2_  = alphaLow2;
  alphaHigh2_ = alphaHigh2;

  gammaLow3_  = gammaLow3;
  gammaHigh3_ = gammaHigh3;
  alphaLow3_  = alphaLow3;
  alphaHigh3_ = alphaHigh3;
}

ESEEIntercalibConstants::~ESEEIntercalibConstants() {

}
