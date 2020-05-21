
#include "CalibFormats/SiStripObjects/interface/SiStripCcu.h"
#include <iostream>

// -----------------------------------------------------------------------------
//
SiStripCcu::SiStripCcu(const FedChannelConnection &conn) : ccuAddr_(conn.ccuAddr()), modules_() {
  modules_.reserve(32);
  addDevices(conn);
}

// -----------------------------------------------------------------------------
//
void SiStripCcu::addDevices(const FedChannelConnection &conn) {
  auto imod = modules_.begin();
  while (imod != modules_.end() && (*imod).ccuChan() != conn.ccuChan()) {
    imod++;
  }
  if (imod == modules_.end()) {
    modules_.emplace_back(conn);
  } else {
    imod->addDevices(conn);
  }
}
