
#include "CalibFormats/SiStripObjects/interface/SiStripFec.h"
#include <iostream>

// -----------------------------------------------------------------------------
//
SiStripFec::SiStripFec(const FedChannelConnection &conn) : fecSlot_(conn.fecSlot()), rings_() {
  rings_.reserve(8);
  addDevices(conn);
}

// -----------------------------------------------------------------------------
//
void SiStripFec::addDevices(const FedChannelConnection &conn) {
  auto iring = rings_.begin();
  while (iring != rings_.end() && (*iring).fecRing() != conn.fecRing()) {
    iring++;
  }
  if (iring == rings_.end()) {
    rings_.emplace_back(conn);
  } else {
    iring->addDevices(conn);
  }
}
