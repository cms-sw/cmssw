#include "DataFormats/L1THGCal/interface/HGCalConcentratorData.h"

using namespace l1t;

HGCalConcentratorData::HGCalConcentratorData(uint32_t data, uint32_t index, uint32_t detid)
    : data_(data), index_(index), detid_(DetId(detid)) {}

HGCalConcentratorData::~HGCalConcentratorData() {}
