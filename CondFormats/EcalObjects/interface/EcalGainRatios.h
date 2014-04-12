#ifndef CondFormats_EcalObjects_EcalGainRatios_H
#define CondFormats_EcalObjects_EcalGainRatios_H
/**
 * Author: Shahram Rahatlou, University of Rome & INFN
 * Created: 22 Feb 2006
 * $Id: EcalGainRatios.h,v 1.3 2007/06/29 06:51:37 innocent Exp $
 **/
#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"
#include "CondFormats/EcalObjects/interface/EcalMGPAGainRatio.h"

typedef EcalCondObjectContainer<EcalMGPAGainRatio> EcalGainRatioMap;
typedef EcalGainRatioMap EcalGainRatios;

#endif
