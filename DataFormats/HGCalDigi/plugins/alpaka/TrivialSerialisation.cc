#include "DataFormats/HGCalDigi/interface/alpaka/HGCalDigiDevice.h"
#include "DataFormats/HGCalDigi/interface/alpaka/HGCalECONDPacketInfoDevice.h"
#include "DataFormats/HGCalDigi/interface/alpaka/HGCalFEDPacketInfoDevice.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"

DEFINE_TRIVIAL_SERIALISER_PLUGIN_DEVICE(hgcaldigi::HGCalDigiDevice);
DEFINE_TRIVIAL_SERIALISER_PLUGIN_DEVICE(hgcaldigi::HGCalECONDPacketInfoDevice);
DEFINE_TRIVIAL_SERIALISER_PLUGIN_DEVICE(hgcaldigi::HGCalFEDPacketInfoDevice);
