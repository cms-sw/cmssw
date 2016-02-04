#include "CondFormats/ESObjects/interface/ESADCToGeVConstant.h"

ESADCToGeVConstant::ESADCToGeVConstant() 
{
  ESvaluelow_=0.;
  ESvaluehigh_=0.;
}

ESADCToGeVConstant::ESADCToGeVConstant(const float & ESvaluelow, const float & ESvaluehigh) {
  ESvaluelow_ = ESvaluelow;
  ESvaluehigh_ = ESvaluehigh;

}

ESADCToGeVConstant::~ESADCToGeVConstant() {

}
