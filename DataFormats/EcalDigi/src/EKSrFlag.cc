// -*- Mode: C++; c-basic-offset: 2; indent-tabs-mode: t; tab-width: 8; -*-
// $Id: EKSrFlag.cc,v 1.1 2014/04/02 10:46:03 shervin Exp $

#include "DataFormats/EcalDigi/interface/EKSrFlag.h"

std::ostream& operator<<(std::ostream& s, const EKSrFlag& digi) {
  s << digi.id() << " "<< digi.flagName() << "(0x"
    << digi.value() << ")";
  return s;
}

