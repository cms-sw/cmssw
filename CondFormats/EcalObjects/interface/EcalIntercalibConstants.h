#ifndef CondFormats_EcalObjects_EcalIntercalibConstants_H
#define CondFormats_EcalObjects_EcalIntercalibConstants_H
/**
 * Author: Shahram Rahatlou, University of Rome & INFN
 * Created: 22 Feb 2006
 * $Id: EcalIntercalibConstants.h,v 1.2 2006/02/23 16:56:34 rahatlou Exp $
 **/
#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"

typedef float EcalIntercalibConstant;
typedef EcalCondObjectContainer<EcalIntercalibConstant> EcalIntercalibConstantMap;
typedef EcalIntercalibConstantMap EcalIntercalibConstants;

#endif
