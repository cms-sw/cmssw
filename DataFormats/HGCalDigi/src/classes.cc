#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"
#include "DataFormats/HGCalDigi/interface/HGCalSimHitAccumHost.h"
#include "DataFormats/HGCalDigi/interface/HGCalDigiHost.h"
#include "DataFormats/HGCalDigi/interface/HGCalECONDPacketInfoHost.h"
#include "DataFormats/HGCalDigi/interface/HGCalFEDPacketInfoHost.h"

SET_PORTABLEHOSTCOLLECTION_READ_RULES(hgcaldigi::HGCalSimHitAccumHost);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(hgcaldigi::HGCalROCDigisAccumHost);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(hgcaldigi::HGCalDigiHost);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(hgcaldigi::HGCalECONDPacketInfoHost);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(hgcaldigi::HGCalFEDPacketInfoHost);
