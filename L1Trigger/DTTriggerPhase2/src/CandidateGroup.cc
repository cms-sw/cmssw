#include "L1Trigger/DTTriggerPhase2/interface/CandidateGroup.h"

#include <iostream>
#include <memory>

using namespace dtbayesam;

//------------------------------------------------------------------
//--- Constructors and destructor
//------------------------------------------------------------------
CandidateGroup::CandidateGroup(DTPatternPtr p) {
  nhits_ = 0;
  nLayerhits_ = 0;
  nisGood_ = 0;
  nLayerDown_ = 0;
  nLayerUp_ = 0;
  pattern_ = p;
}

CandidateGroup::CandidateGroup() {}

CandidateGroup::~CandidateGroup() {}

void CandidateGroup::addHit(DTPrimitive dthit, int lay, bool isGood) {
  //Add a hit, check if the hits layer was fired and if it wasn't add it to the fired layers
  candHits_.push_back(std::make_shared<DTPrimitive>(dthit));
  if (quality_ != (quality_ | qualitybits(std::pow(2, lay))))
    nLayerhits_++;
  if (isGood)
    nisGood_++;
  quality_ = quality_ | qualitybits(std::pow(2, lay));
  nhits_++;
  if (lay <= 3)
    nLayerDown_++;
  if (lay >= 4)
    nLayerUp_++;
}

void CandidateGroup::removeHit(DTPrimitive dthit) {
  //Add a hit, check if the hits layer was fired and if it wasn't add it to the fired layers
  DTPrimitivePtrs tempHits;
  nhits_ = 0;
  nLayerDown_ = 0;
  nLayerUp_ = 0;
  nLayerhits_ = 0;
  nisGood_ = 0;
  quality_ = qualitybits("00000000");
  for (auto dt_it = candHits_.begin(); dt_it != candHits_.end(); dt_it++) {
    if (dthit.layerId() == (*dt_it)->layerId() && dthit.channelId() == (*dt_it)->channelId()) {
    } else {
      if (pattern_->latHitIn((*dt_it)->layerId(), (*dt_it)->channelId(), 0) > -5)
        nisGood_++;
      if (quality_ != (quality_ | qualitybits(std::pow(2, (*dt_it)->layerId()))))
        nLayerhits_++;
      quality_ = quality_ | qualitybits(std::pow(2, (*dt_it)->layerId()));
      nhits_++;
      if ((*dt_it)->layerId() <= 3)
        nLayerDown_++;
      else if ((*dt_it)->layerId() >= 4)
        nLayerUp_++;
      tempHits.push_back(*dt_it);
    }
  }
  candHits_ = tempHits;
}

bool CandidateGroup::operator>(const CandidateGroup& cOther) const {
  //First number of good (in pattern) matched hits
  if (nisGood_ > cOther.nisGood())
    return true;
  else if (nisGood_ < cOther.nisGood())
    return false;
  //Tehn quality_ is number of layers fired
  else if (nLayerhits_ > cOther.nLayerhits())
    return true;
  else if (nLayerhits_ < cOther.nLayerhits())
    return false;
  //Then number of matched hits (if multiple hits in a layer is better)
  else if (nhits_ > cOther.nhits())
    return true;
  else if (nhits_ < cOther.nhits())
    return false;
  //Balanced quality_, prefer 3+2 over 4+1 for example
  else if ((nLayerUp_ - nLayerDown_) > (cOther.nLayerUp() - cOther.nLayerDown()))
    return true;
  else if ((nLayerUp_ - nLayerDown_) < (cOther.nLayerUp() - cOther.nLayerDown()))
    return false;
  //Last, patterns with less gen hits are better matched
  else if (pattern_->genHits().size() < (cOther.pattern()->genHits().size()))
    return true;
  else if (pattern_->genHits().size() > (cOther.pattern()->genHits().size()))
    return false;
  //For a total order relation we need this additional dummy
  else if (candId_ < cOther.candId())
    return true;
  else if (candId_ > cOther.candId())
    return false;

  //If we are here, they are basically equal so it doesn't matter who goes first
  return true;
}

bool CandidateGroup::operator==(const CandidateGroup& cOther) const {
  //First number of good hits
  if (nisGood_ != cOther.nisGood())
    return false;
  //First quality_ is number of layers fired
  else if (nLayerhits_ != cOther.nLayerhits())
    return false;
  //Then number of matched hits (if multiple hits in a layer is better)
  else if (nhits_ != cOther.nhits())
    return false;
  //Balanced quality_, prefer 3+2 over 4+1 for example
  else if ((nLayerUp_ - nLayerDown_) != (cOther.nLayerUp() - cOther.nLayerDown()))
    return false;
  //Last, patterns with less gen hits are better matched
  return true;
}
