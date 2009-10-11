#include "CondFormats/ESObjects/interface/ESEEIntercalibConstants.h"

ESEEIntercalibConstants::ESEEIntercalibConstants() 
{
  gammaLow_=0.;
  gammaHigh_=0.;
  alphaLow_=0.;
  alphaHigh_=0.;
}

ESEEIntercalibConstants::ESEEIntercalibConstants(const float & gammaLow, const float & alphaLow, const float & gammaHigh, const float & alphaHigh) {
  gammaLow_ = gammaLow;
  gammaHigh_ = gammaHigh;
  alphaLow_ = alphaLow;
  alphaHigh_ = alphaHigh;
}

ESEEIntercalibConstants::~ESEEIntercalibConstants() {

}
