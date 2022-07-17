#include "L1Trigger/DTTriggerPhase2/interface/DTPattern.h"
#include "L1Trigger/DTTriggerPhase2/interface/constants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

//------------------------------------------------------------------
//--- Constructors and destructor
//------------------------------------------------------------------
DTPattern::DTPattern() {}

DTPattern::DTPattern(RefDTPatternHit seedUp, RefDTPatternHit seedDown) : seedUp_(seedUp), seedDown_(seedDown) {
  //On creation, pattern is based on seeds, with no hits. Due to traslational simmetry we only need the superlayer indexes as well as the cell index difference
  id_ = std::make_tuple(std::get<0>(seedUp), std::get<0>(seedDown), std::get<1>(seedUp) - std::get<1>(seedDown));
  if (debug_)
    LogDebug("DTPattern") << "Pattern id: " << std::get<0>(id_) << " , " << std::get<1>(id_) << " , "
                          << std::get<2>(id_);
}

DTPattern::DTPattern(int SL1, int SL2, int diff) {
  //On creation, pattern is based on seeds, with no hits. Due to traslational simmetry we only need the superlayer indexes as well as the cell index difference
  seedUp_ = std::make_tuple(SL1, 0, 0);
  seedDown_ = std::make_tuple(SL2, diff, 0);
  id_ = std::make_tuple(SL1, SL2, diff);
  if (debug_)
    LogDebug("DTPattern") << "Pattern id: " << std::get<0>(id_) << " , " << std::get<1>(id_) << " , "
                          << std::get<2>(id_);
}

void DTPattern::addHit(RefDTPatternHit hit) {
  //Add additional gen level hits in the gen pattern coordinates (untranslated)
  genHits_.push_back(hit);
  if (debug_)
    LogDebug("DTPattern") << "Added gen hit: " << std::get<0>(hit) << " , " << std::get<1>(hit) << " , "
                          << std::get<2>(hit);
}

int DTPattern::latHitIn(int slId, int chId, int allowedVariance) const {
  //Check if a hit is inside of the pattern for a given pattern width
  int temp = -999;
  for (auto it = this->genHits_.begin(); it != this->genHits_.end(); ++it) {
    if (slId == (std::get<0>(*it) - 1)) {
      if (chId == (std::get<1>(*it) + recoseedDown_)) {
        return std::get<2>(*it);
      }
      //This is equivalent to an allowed discrete width of the pattern (configured)
      else if ((chId <= (std::get<1>(*it) + recoseedDown_ + allowedVariance)) &&
               (chId >= (std::get<1>(*it) + recoseedDown_ - allowedVariance))) {
        temp = -10;
      }
    }
  }
  return temp;
}

std::ostream &operator<<(std::ostream &out, DTPattern const &p) {
  //Friend for printing pattern information trough iostream
  out << "Pattern id: " << std::get<0>(p.id()) << " , " << std::get<1>(p.id()) << " , " << std::get<2>(p.id())
      << std::endl;
  std::vector<RefDTPatternHit> thegenHits = p.genHits();
  out << "Pattern hits: " << std::endl;

  for (std::vector<RefDTPatternHit>::iterator itHit = thegenHits.begin(); itHit != thegenHits.end(); itHit++) {
    out << "[" << std::get<0>(*itHit) << " , " << std::get<1>(*itHit) << " , " << std::get<2>(*itHit) << "]";
  }
  return out;
}

DTPattern::~DTPattern() {}
