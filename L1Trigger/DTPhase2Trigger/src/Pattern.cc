#include "L1Trigger/DTPhase2Trigger/interface/Pattern.h"
#include "L1Trigger/DTPhase2Trigger/interface/constants.h"
#include <iostream> 

//------------------------------------------------------------------
//--- Constructores y destructores
//------------------------------------------------------------------
Pattern::Pattern() {
}

Pattern::Pattern(RefPatternHit seedUp, RefPatternHit seedDown) {
  //On creation, pattern is based on seeds, with no hits. Due to traslational simmetry we only need the superlayer indexes as well as the cell index difference
  seedUp   = seedUp;
  seedDown = seedDown;
  id = std::make_tuple(std::get<0>(seedUp), std::get<0>(seedDown), std::get<1>(seedUp)-std::get<1>(seedDown));
  if (debug) std::cout << "Pattern id: " << std::get<0>(id) << " , " << std::get<1>(id) << " , " << std::get<2>(id) << std::endl; 
}

Pattern::Pattern(int SL1, int SL2, int diff) {
  //On creation, pattern is based on seeds, with no hits. Due to traslational simmetry we only need the superlayer indexes as well as the cell index difference
  seedUp   = std::make_tuple(SL1, 0, 0);
  seedDown = std::make_tuple(SL2, diff, 0);
  id = std::make_tuple(SL1,SL2,diff);
  if (debug) std::cout << "Pattern id: " << std::get<0>(id) << " , " << std::get<1>(id) << " , " << std::get<2>(id) << std::endl; 
}

void Pattern::AddHit(RefPatternHit hit){
  //Add additional gen level hits in the gen pattern coordinates (untranslated)
  genHits.push_back(hit);
  if (debug) std::cout << "Added gen hit: " << std::get<0>(hit) << " , " << std::get<1>(hit) << " , " << std::get<2>(hit) << std::endl; 
}

int Pattern::LatHitIn(int slId, int chId, int allowedVariance){
  //Check if a hit is inside of the pattern for a given pattern width
  int temp = -999;
  //std::cout << std::endl << "Compare " << slId << " , " << chId << " with" <<  std::endl;
  for (std::vector<RefPatternHit>::iterator it = genHits.begin() ; it != genHits.end(); ++it){
    //std::cout << std::get<0>(*it)-1 << " , " << std::get<1>(*it) + recoseedDown << std::endl;
    if (slId == (std::get<0>(*it)-1)){
      if (chId == (std::get<1>(*it) + recoseedDown)){
        return std::get<2>(*it);
      }
      //This is equivalent to an allowed discrete width of the pattern (configured) 
      else if ((chId <= (std::get<1>(*it) + recoseedDown + allowedVariance )) && (chId >= (std::get<1>(*it) + recoseedDown - allowedVariance )) ){
        temp = -10;
      }
    }
  }
  return temp;
}

std::ostream & operator << (std::ostream &out, Pattern &p)
{
    //Friend for printing pattern information trough iostream
    out << "Pattern id: " << std::get<0>(p.GetId()) << " , " << std::get<1>(p.GetId()) << " , " << std::get<2>(p.GetId()) << std::endl;
    std::vector<RefPatternHit> thegenHits = p.GetGenHits();
    out << "Pattern hits: ";

    for (std::vector<RefPatternHit>::iterator itHit = thegenHits.begin(); itHit != thegenHits.end(); itHit++){
         out << "[" << std::get<0>(*itHit) << " , " << std::get<1>(*itHit) << " , " << std::get<2>(*itHit) << "]";
    } 
    return out;
}

Pattern::~Pattern() {
}
