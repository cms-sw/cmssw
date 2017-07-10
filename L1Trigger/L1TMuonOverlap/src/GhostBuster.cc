#include "L1Trigger/L1TMuonOverlap/interface/GhostBuster.h"

#include <sstream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFConfiguration.h"

namespace { 
  int phiGMT(int phiAlgo) { return phiAlgo*437/pow(2,12); }
}

void OMTFGhostBuster::select(std::vector<AlgoMuon> & refHitCands, int charge){

  std::vector<AlgoMuon> refHitCleanCands;
  // Sort candidates with decreased goodness,
  // where goodness definied in < operator of AlgoMuon
  std::sort( refHitCands.rbegin(), refHitCands.rend() );

  for(auto & refHitCand : refHitCands){
    bool isGhost=false;
    for(auto & refHitCleanCand : refHitCleanCands){
      //do not accept candidates with similar phi (any charge combination)
      //veto window 5deg(=half of logic cone)=5/360*5760=80"logic strips" 
      //veto window 5 degree in GMT scale is 5/360*576=8 units
      if (std::abs( phiGMT(refHitCand.getPhi()) - phiGMT(refHitCleanCand.getPhi()) ) < 8 ) { 
//      if(std::abs(it1->getPhi() - it2->getPhi())<5/360.0*nPhiBins){
        isGhost=true;
        break;
      }
    }
    if(refHitCand.getQ()>0 && !isGhost) refHitCleanCands.push_back(refHitCand);
  }

  refHitCleanCands.resize( 3, AlgoMuon(0,999,9999,0,0,0,0,0) );

  std::stringstream myStr;
  bool hasCandidates = false;
  for(auto & refHitCand : refHitCands){
    if(refHitCand.getQ()){
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
