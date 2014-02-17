#ifndef CondFormats_EcalObjects_EcalChannelStatus_H
#define CondFormats_EcalObjects_EcalChannelStatus_H
/**
 * Author: Paolo Meridiani
 * Created: 14 November 2006
 * $Id: EcalChannelStatus.h,v 1.2 2007/09/27 09:42:55 ferriff Exp $
 **/

#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatusCode.h"

typedef EcalCondObjectContainer<EcalChannelStatusCode> EcalChannelStatusMap;
typedef EcalChannelStatusMap EcalChannelStatus;

#endif
