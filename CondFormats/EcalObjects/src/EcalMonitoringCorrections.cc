/**
 * Author: Paolo Meridiani
 * Created: 14 Nov 2006
 * $Id: $
 **/

#include "CondFormats/EcalObjects/interface/EcalMonitoringCorrections.h"

EcalMonitoringCorrections::EcalMonitoringCorrections() {
}

EcalMonitoringCorrections::~EcalMonitoringCorrections() {

}

void
EcalMonitoringCorrections::setValue(const uint32_t& id, const EcalMonitoringCorrection& value) {
  map_[id] = value;
}
