#ifndef CondFormats_EcalObjects_EcalChannelStatus_H
#define CondFormats_EcalObjects_EcalChannelStatus_H
/**
 * Author: Paolo Meridiani
 * Created: 14 November 2006
 * $Id: EcalChannelStatus.h,v 1.1 2006/11/16 18:18:24 meridian Exp $
 **/

#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatusCode.h"

typedef EcalCondObjectContainer<EcalChannelStatusCode> EcalChannelStatusMap;
typedef EcalChannelStatusMap EcalChannelStatus;

#endif
