#ifndef CondFormats_EcalObjects_EcalWeightXtalGroups_H
#define CondFormats_EcalObjects_EcalWeightXtalGroups_H
/**
 * Author: Shahram Rahatlou, University of Rome & INFN
 * Created: 22 Feb 2006
 * $Id: EcalWeightXtalGroups.h,v 1.3 2007/06/29 07:04:30 innocent Exp $
 **/

#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"
#include "CondFormats/EcalObjects/interface/EcalXtalGroupId.h"

typedef EcalCondObjectContainer<EcalXtalGroupId> EcalWeightXtalGroups;
typedef EcalWeightXtalGroups EcalXtalGroupsMap;

#endif
