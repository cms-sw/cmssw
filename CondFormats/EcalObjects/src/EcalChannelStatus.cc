/**
 * Author: Paolo Meridiani
 * Created: 14 Nov 2006
 * $Id: $
 **/

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

EcalChannelStatus::EcalChannelStatus() {
}

EcalChannelStatus::~EcalChannelStatus() {

}

void
EcalChannelStatus::setValue(const uint32_t& id, const EcalChannelStatusCode& value) {
  map_[id] = value;
}
