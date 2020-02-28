#include "CondFormats/PhysicsToolsObjects/interface/PhysicsPerformancePayload.h"

//#include <iostream>

int PhysicsPerformancePayload::nRows() const { return table_.size() / stride_; }

PhysicsPerformancePayload::Row PhysicsPerformancePayload::getRow(int n) const {
  Row temp;
  copy(table_.begin() + (n * stride_), table_.begin() + (n + 1) * stride_, back_inserter(temp));
  return temp;
}

PhysicsPerformancePayload::PhysicsPerformancePayload(int stride, const std::vector<float>& table)
    : stride_(stride), table_(table) {}

#include "FWCore/Utilities/interface/typelookup.h"

TYPELOOKUP_DATA_REG(PhysicsPerformancePayload);
