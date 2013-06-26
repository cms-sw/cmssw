#ifndef RecoEcal_EcalNextToDeadChannel_H
#define RecoEcal_EcalNextToDeadChannel_H

/** Define an ecal container to tell wether the channel is close to a dead one.
    If the value is "true", the channel is next to a dead one
    $Id: EcalNextToDeadChannel.h,v 1.1 2011/05/19 14:38:28 argiro Exp $
**/
 
#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"

typedef EcalCondObjectContainer<uint8_t> EcalNextToDeadChannel;

#endif
