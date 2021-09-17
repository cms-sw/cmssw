// -*- Mode: C++; c-basic-offset: 2; indent-tabs-mode: t; tab-width: 8; -*-

#include "DataFormats/EcalDigi/interface/EESrFlag.h"

std::ostream& operator<<(std::ostream& s, const EESrFlag& digi) {
  s << digi.id() << " " << digi.flagName() << "(0x" << digi.value() << ")";
  return s;
}
