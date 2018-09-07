#include "L1Trigger/L1TMuonOverlap/interface/GhostBusterPreferRefDt.h"

#include <sstream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFConfiguration.h"

namespace { 
  int phiGMT(int phiAlgo) { return phiAlgo*437/pow(2,12); }
}

std::vector<AlgoMuon> GhostBusterPreferRefDt::select(std::vector<AlgoMuon> refHitCands, int charge) {
  auto customLess = [&](const AlgoMuon& a, const AlgoMuon& b)->bool {
    int aRefLayerLogicNum = omtfConfig->getRefToLogicNumber()[a.getRefLayer()];
    int bRefLayerLogicNum = omtfConfig->getRefToLogicNumber()[b.getRefLayer()];
    if(a.getQ() > b.getQ())
      return false;
    else if(a.getQ()==b.getQ() && aRefLayerLogicNum < bRefLayerLogicNum) {
      return false;
    }
    else if (a.getQ()==b.getQ() && aRefLayerLogicNum == bRefLayerLogicNum && a.getDisc() > b.getDisc() )
      return false;
    else if (a.getQ()==b.getQ() && aRefLayerLogicNum == bRefLayerLogicNum && a.getDisc() == b.getDisc() && a.getPatternNumber() > b.getPatternNumber() )
      return false;
    else if (a.getQ()==b.getQ() && aRefLayerLogicNum == bRefLayerLogicNum && a.getDisc() == b.getDisc() && a.getPatternNumber() == b.getPatternNumber() && a.getRefHitNumber() < b.getRefHitNumber())
      return false;
    else
      return true;
    //TODO check if the firmware really includes the pattern number and refHit number here
  };

/*  auto customLess = [&](const AlgoMuon& a, const AlgoMuon& b)->bool {
    int aRefLayerLogicNum = omtfConfig->getRefToLogicNumber()[a.getRefLayer()];
    int bRefLayerLogicNum = omtfConfig->getRefToLogicNumber()[b.getRefLayer()];
    if(a.getQ() > b.getQ())
      return false;
    else if(a.getQ()==b.getQ() && aRefLayerLogicNum < bRefLayerLogicNum) {
      return false;
    }
    else if (a.getQ()==b.getQ() && aRefLayerLogicNum == bRefLayerLogicNum && a.getPatternNumber() > b.getPatternNumber() )
      return false;
    else if (a.getQ()==b.getQ() && aRefLayerLogicNum == bRefLayerLogicNum && a.getPatternNumber() == b.getPatternNumber() && a.getRefHitNumber() < b.getRefHitNumber())
      return false;
    else
      return true;
    //TODO check if the firmware really includes the pattern number and refHit number here
  };*/

  std::vector<AlgoMuon> refHitCleanCands;
  // Sort candidates with decreased goodness,
  // where goodness definied in < operator of AlgoMuon
  std::sort( refHitCands.rbegin(), refHitCands.rend(), customLess );

  for(std::vector<AlgoMuon>::iterator it1 = refHitCands.begin();
      it1 != refHitCands.end(); ++it1){
    bool isGhost=false;
    for(std::vector<AlgoMuon>::iterator it2 = refHitCleanCands.begin();
    it2 != refHitCleanCands.end(); ++it2){
      //do not accept candidates with similar phi (any charge combination)
      //veto window 5deg(=half of logic cone)=5/360*5760=80"logic strips" 
      //veto window 5 degree in GMT scale is 5/360*576=8 units
      if (std::abs( phiGMT(it1->getPhi()) - phiGMT(it2->getPhi()) ) < 8 ) { 
//      if(std::abs(it1->getPhi() - it2->getPhi())<5/360.0*nPhiBins){
        isGhost=true;
        break;
        //which one candidate is killed depends only on the order in the refHitCands (the one with smaller index is taken), and this order is assured by the sort above
        //TODO here the candidate that is killed does not kill other candidates - check if the firmware does the same (KB)
      }
    }
    if(it1->getQ()>0 && !isGhost) refHitCleanCands.push_back(*it1);
  }

  refHitCleanCands.resize( 3, AlgoMuon(0,999,9999,0,0,0,0,0) ); //FIXME
  //refHitCleanCands.resize( 3, AlgoMuon() );


  std::stringstream myStr;
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
  return refHitCleanCands;
}
