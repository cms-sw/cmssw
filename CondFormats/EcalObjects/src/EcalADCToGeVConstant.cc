/**
 * Author: Shahram Rahatlou, University of Rome & INFN
 * Created: 22 Feb 2006
 * $Id: $
 **/

#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"

EcalADCToGeVConstant::EcalADCToGeVConstant() {
  value_ = 0;
}

EcalADCToGeVConstant::EcalADCToGeVConstant(const float & value) {
  value_ = value;
}

EcalADCToGeVConstant::~EcalADCToGeVConstant() {

}
