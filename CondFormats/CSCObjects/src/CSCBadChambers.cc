#include "CondFormats/CSCObjects/interface/CSCBadChambers.h"

CSCBadChambers::CSCBadChambers(){}

CSCBadChambers::~CSCBadChambers(){}

std::vector<int> CSCBadChambers::chambers() const {
  return theChambers;
}
