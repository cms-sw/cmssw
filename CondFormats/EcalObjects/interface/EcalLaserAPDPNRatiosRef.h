#ifndef CondFormats_EcalObjects_EcalLaserAPDPNRatiosRef_H
#define CondFormats_EcalObjects_EcalLaserAPDPNRatiosRef_H
/**
 * Author: Vladlen Timciuc, Caltech, Pasadena, USA
 * Created: 10 July 2007
 * $Id: EcalLaserAPDPNRatiosRef.h,v 1.3 2007/09/09 12:51:14 torimoto Exp $
 **/

#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"

typedef float EcalLaserAPDPNref;
typedef EcalCondObjectContainer<EcalLaserAPDPNref> EcalLaserAPDPNRatiosRefMap;
typedef EcalLaserAPDPNRatiosRefMap EcalLaserAPDPNRatiosRef;

#endif
