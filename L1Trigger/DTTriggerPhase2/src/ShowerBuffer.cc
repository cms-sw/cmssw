#include "L1Trigger/DTTriggerPhase2/interface/ShowerBuffer.h"

#include <cmath>
#include <iostream>
#include <memory>

using namespace cmsdt;

ShowerBuffer::ShowerBuffer() {
  nprimitives_ = 0;
  shower_flag_ = false;
}

void ShowerBuffer::addHit(DTPrimitive &prim){
  prim_.push_back(std::make_shared<DTPrimitive>(prim));
  nprimitives_++;
}

