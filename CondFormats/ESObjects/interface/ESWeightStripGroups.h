#ifndef CondFormats_ESObjects_ESWeightStripGroups_H
#define CondFormats_ESObjects_ESWeightStripGroups_H
/**
 * Author: Shahram Rahatlou, University of Rome & INFN
 * Created: 22 Feb 2006
 * $Id: ESWeightStripGroups.h,v 1.1 2009/03/27 15:31:48 fra Exp $
 **/

#include "CondFormats/ESObjects/interface/ESCondObjectContainer.h"
#include "CondFormats/ESObjects/interface/ESStripGroupId.h"

typedef ESCondObjectContainer<ESStripGroupId> ESWeightStripGroups;
typedef ESWeightStripGroups ESStripGroupsMap;

#endif
