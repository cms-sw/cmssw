#include "L1Trigger/DTPhase2Trigger/interface/CandidateGroup.h"
#include <iostream> 

//------------------------------------------------------------------
//--- Constructores y destructores
//------------------------------------------------------------------
CandidateGroup::CandidateGroup(Pattern* p) {
  nhits = 0;
  nLayerhits = 0;
  nisGood = 0;
  nLayerDown = 0;
  nLayerUp = 0;
  pattern = p;
}

CandidateGroup::CandidateGroup() {
}

CandidateGroup::~CandidateGroup() {
}

void CandidateGroup::AddHit(DTPrimitive dthit, int lay, bool isGood){
  //Add a hit, check if the hits layer was fired and if it wasn't add it to the fired layers
  candHits.push_back(dthit);
  if (quality != (quality | std::bitset<8>(power(2, lay)))) nLayerhits++;
  if (isGood) nisGood++;
  quality = quality | std::bitset<8>(power(2, lay));
  nhits++;
  if (lay <=3) nLayerDown++;
  if (lay >=4) nLayerUp++;
}

void CandidateGroup::RemoveHit(DTPrimitive dthit){
  //Add a hit, check if the hits layer was fired and if it wasn't add it to the fired layers
  std::vector<DTPrimitive> tempHits;    
  nhits = 0;
  nLayerDown = 0;
  nLayerUp = 0;
  nLayerhits = 0;
  nisGood = 0;
  quality = std::bitset<8>("00000000");
  //std::cout << "Removing hit ";
  //std::cout << dthit.getLayerId() << " , " << dthit.getChannelId() << std::endl;
  for (std::vector<DTPrimitive>::iterator dt_it = candHits.begin(); dt_it != candHits.end(); dt_it++){ 
    if (dthit.getLayerId() == dt_it->getLayerId() && dthit.getChannelId() == dt_it->getChannelId()){
      //std::cout << "Found hit to remove" << std::endl;
    }
    else{
      //std::cout << "Redoing quality " << nisGood << " , " << nLayerhits << " , " << nhits << std::endl;
      if (pattern->LatHitIn(dt_it->getLayerId(), dt_it->getChannelId(), 0) > -5) nisGood++;
      if (quality != (quality | std::bitset<8>(power(2, dt_it->getLayerId())))) nLayerhits++;
      quality = quality | std::bitset<8>(power(2, dt_it->getLayerId()));
      nhits++;
      if (dt_it->getLayerId() <=3) nLayerDown++;
      else if (dt_it->getLayerId() >=4) nLayerUp++;
      tempHits.push_back(*dt_it);
    }
  }
  candHits = tempHits;
}


bool CandidateGroup::operator> (const CandidateGroup& cOther) const{
  //First number of good (in pattern) matched hits
  if      (nisGood > cOther.getNisGood()) return true;
  else if (nisGood < cOther.getNisGood()) return false;
  //Tehn quality is number of layers fired  
  else if (nLayerhits > cOther.getNLayerhits()) return true;
  else if (nLayerhits < cOther.getNLayerhits()) return false;
  //Then number of matched hits (if multiple hits in a layer is better)
  else if (nhits > cOther.getNhits()) return true;
  else if (nhits < cOther.getNhits()) return false;
  //Balanced quality, prefer 3+2 over 4+1 for example
  else if ((nLayerUp - nLayerDown) > (cOther.getNLayerUp() - cOther.getNLayerDown())) return true;
  else if ((nLayerUp - nLayerDown) < (cOther.getNLayerUp() - cOther.getNLayerDown())) return false;
  //Last, patterns with less gen hits are better matched
  else if (pattern->GetGenHits().size() < (cOther.getPattern()->GetGenHits().size())) return true;
  else if (pattern->GetGenHits().size() > (cOther.getPattern()->GetGenHits().size()))  return false;
  //For a total order relation we need this additional dummy
  else if (candId < cOther.getCandId()) return true;
  else if (candId > cOther.getCandId()) return false;
  
  //If we are here, they are basically equal so it doesn't matter who goes first
  return true;
}


bool CandidateGroup::operator== (const CandidateGroup& cOther) const{
  //First number of good hits
  if (nisGood != cOther.getNisGood()) return false;
  //First quality is number of layers fired
  else if (nLayerhits != cOther.getNLayerhits()) return false;
  //Then number of matched hits (if multiple hits in a layer is better)
  else if (nhits != cOther.getNhits()) return false;
  //Balanced quality, prefer 3+2 over 4+1 for example
  else if ((nLayerUp - nLayerDown) != (cOther.getNLayerUp() - cOther.getNLayerDown())) return false;
  //Last, patterns with less gen hits are better matched
  return true;
}

