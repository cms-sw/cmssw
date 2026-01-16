#include "DataFormats/HGCalDigi/interface/HGCalDigiHost.h"
#include "DataFormats/HGCalDigi/interface/HGCalECONDPacketInfoHost.h"
#include "DataFormats/HGCalDigi/interface/HGCalFEDPacketInfoHost.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/SerialiserFactory.h"

DEFINE_TRIVIAL_SERIALISER_PLUGIN(hgcaldigi::HGCalDigiHost);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(hgcaldigi::HGCalECONDPacketInfoHost);
DEFINE_TRIVIAL_SERIALISER_PLUGIN(hgcaldigi::HGCalFEDPacketInfoHost);
