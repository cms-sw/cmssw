#include "DataFormats/HGCalDigi/interface/HGCalDigiHost.h"
#include "DataFormats/HGCalDigi/interface/HGCalECONDPacketInfoHost.h"
#include "DataFormats/HGCalDigi/interface/HGCalFEDPacketInfoHost.h"
#include "DataFormats/HGCalDigi/interface/alpaka/HGCalDigiDevice.h"
#include "DataFormats/HGCalDigi/interface/alpaka/HGCalECONDPacketInfoDevice.h"
#include "DataFormats/HGCalDigi/interface/alpaka/HGCalFEDPacketInfoDevice.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"

DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(hgcaldigi::HGCalDigiHost, hgcaldigi::HGCalDigiDevice);
DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(hgcaldigi::HGCalECONDPacketInfoHost, hgcaldigi::HGCalECONDPacketInfoDevice);
DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN(hgcaldigi::HGCalFEDPacketInfoHost, hgcaldigi::HGCalFEDPacketInfoDevice);
