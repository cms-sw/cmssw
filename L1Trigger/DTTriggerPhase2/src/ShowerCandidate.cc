#include "L1Trigger/DTTriggerPhase2/interface/ShowerCandidate.h"

#include <cmath>
#include <iostream>
#include <memory>

using namespace cmsdt;

ShowerCandidate::ShowerCandidate() {
  nhits_ = 0;
  avgTime_ = 0;
  avgPos_ = 0;
}

ShowerCandidate::ShowerCandidate(ShowerBuffer& buff){
  nhits_ = buff.getNhits();
  rawId_ = buff.getRawId();
  for ( auto& hit : buff.getHits() ){
    hits_.push_back(hit);
  }
}

ShowerCandidate::ShowerCandidate(ShowerBufferPtr& buffPtr){
  nhits_ = buffPtr->getNhits();
  rawId_ = buffPtr->getRawId();
  for ( auto& hit : buffPtr->getHits() ){
    hits_.push_back(hit);
  }
 
  // Set properties of the shower candidate 
  setAvgTime();
  setAvgPos();
}


void ShowerCandidate::setAvgPos() {
  // Sets the average position in X axis 
  for (auto& hit : hits_) {
      avgPos_ = avgPos_ + hit->wireHorizPos();
  }
  avgPos_ = avgPos_ / nhits_;
}

void ShowerCandidate::setAvgTime() {
  for (auto& hit : hits_) {
      avgTime_ = avgTime_ + hit->tdcTimeStamp();
  }
  avgTime_ = avgTime_ / nhits_;
}
