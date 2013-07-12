#ifndef CondFormats_ESObjects_ESWeightStripGroups_H
#define CondFormats_ESObjects_ESWeightStripGroups_H
/**
 * Author: Shahram Rahatlou, University of Rome & INFN
 * Created: 22 Feb 2006
 * $Id: ESWeightStripGroups.h,v 1.4 2007/09/27 09:42:55 ferriff Exp $
 **/

#include "CondFormats/ESObjects/interface/ESCondObjectContainer.h"
#include "CondFormats/ESObjects/interface/ESStripGroupId.h"

typedef ESCondObjectContainer<ESStripGroupId> ESWeightStripGroups;
typedef ESWeightStripGroups ESStripGroupsMap;

#endif
