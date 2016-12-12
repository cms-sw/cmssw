#include <cmath>
#include <map>
#include <set>
#include <vector>
#include <ostream>
#include <memory>
#include <algorithm>
#include <iostream>
#include <strstream>
#include <bitset>
#include <cassert>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1TMuonOverlap/interface/OMTFConfiguration.h"
#include "L1Trigger/L1TMuonOverlap/interface/GhostBuster.h"

void OMTFGhostBuster::select(std::vector<AlgoMuon> & refHitCands, int charge){

  std::vector<AlgoMuon> refHitCleanCands;
  // Sort candidates with decreased goodness,
  // where goodness definied in < operator of AlgoMuon
  std::sort( refHitCands.rbegin(), refHitCands.rend() );

  for(std::vector<AlgoMuon>::iterator it1 = refHitCands.begin();
      it1 != refHitCands.end(); ++it1){
    bool isGhost=false;
    for(std::vector<AlgoMuon>::iterator it2 = refHitCleanCands.begin();
    it2 != refHitCleanCands.end(); ++it2){
      //do not accept candidates with similar phi (any charge combination)
      //veto window 5deg(=half of logic cone)=5/360*5760=80"logic strips"
      if(std::abs(it1->getPhi() - it2->getPhi())<5/360.0*nPhiBins){
        isGhost=true;
        break;
      }
    }
    if(it1->getQ()>0 && !isGhost) refHitCleanCands.push_back(*it1);
  }

  refHitCleanCands.resize( 3, AlgoMuon(0,999,9999,0,0,0,0,0) );

  std::ostringstream myStr;
  bool hasCandidates = false;
  for(unsigned int iRefHit=0;iRefHit<refHitCands.size();++iRefHit){
    if(refHitCands[iRefHit].getQ()){
      hasCandidates=true;
      break;
    }
  }
  for(unsigned int iRefHit=0;iRefHit<refHitCands.size();++iRefHit){
    if(refHitCands[iRefHit].getQ()) myStr<<"Ref hit: "<<iRefHit<<" "<<refHitCands[iRefHit]<<std::endl;
  }
  myStr<<"Selected Candidates with charge: "<<charge<<std::endl;
  for(unsigned int iCand=0; iCand<refHitCleanCands.size(); ++iCand){
    myStr<<"Cand: "<<iCand<<" "<<refHitCleanCands[iCand]<<std::endl;
  }

  if(hasCandidates) edm::LogInfo("OMTF Sorter")<<myStr.str();

  // update refHitCands with refHitCleanCands 
  refHitCands = refHitCleanCands;
}
