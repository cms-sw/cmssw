#include <cassert>
#include <iostream>
#include <strstream>
#include <algorithm>
#include <bitset>


#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1TMuonOverlap/interface/OMTFConfiguration.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFSorter.h"

#include "L1Trigger/RPCTrigger/interface/RPCConst.h"
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
AlgoMuon OMTFSorter::sortSingleResult(const OMTFResult & aResult){

  OMTFResult::vector1D pdfValsVec = aResult.getSummaryVals();
  OMTFResult::vector1D nHitsVec = aResult.getSummaryHits();
  OMTFResult::vector1D refPhiVec = aResult.getRefPhis();
  OMTFResult::vector1D refEtaVec = aResult.getRefEtas();
  OMTFResult::vector1D hitsVec = aResult.getHitsWord();
  OMTFResult::vector1D refPhiRHitVec = aResult.getRefPhiRHits();

  assert(pdfValsVec.size()==nHitsVec.size());

  unsigned int nHitsMax = 0;
  unsigned int pdfValMax = 0;
  unsigned int hitsWord = 0;
  int refPhi = 1024;
  int refEta = -10;
  int refLayer = -1;
  int refPhiRHit = 1024;

  AlgoMuon  sortedResult(pdfValMax, refPhi, refEta, refLayer, hitsWord, nHitsMax);
  
  ///Find a result with biggest number of hits
  for(auto itHits: nHitsVec){
    if(itHits>nHitsMax) nHitsMax = itHits;
  }

  if(!nHitsMax) return sortedResult;

  for(unsigned int ipdfVal=0;ipdfVal<pdfValsVec.size();++ipdfVal){
    if(nHitsVec[ipdfVal] == nHitsMax){
      if(pdfValsVec[ipdfVal]>pdfValMax){
      	pdfValMax = pdfValsVec[ipdfVal];
      	refPhi = refPhiVec[ipdfVal];
      	refEta = refEtaVec[ipdfVal];
      	refLayer = ipdfVal;
      	hitsWord = hitsVec[ipdfVal];
        refPhiRHit = refPhiRHitVec[ipdfVal];  
      }
    }
  }

  sortedResult.setDisc(pdfValMax);
  sortedResult.setPhi(refPhi);
  sortedResult.setEta(refEta);
  sortedResult.setRefLayer(refLayer);
  sortedResult.setHits(hitsWord);
  sortedResult.setQ(nHitsMax);
  sortedResult.setPhiRHit(refPhiRHit);

  return sortedResult;
}
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
AlgoMuon OMTFSorter::sortRefHitResults(const OMTFProcessor::resultsMap & aResultsMap,
					  int charge){

  unsigned int pdfValMax = 0;
  unsigned int nHitsMax = 0;
  unsigned int hitsWord = 0;
  int refPhi = 9999;
  int refEta = 999;
  int refLayer = -1;
  int refPhiRHit = 9999;
  Key bestKey;

//  std::cout <<" ====== sortRefHitResults: " << std::endl;
  for(auto itKey: aResultsMap){
    if(charge!=0 && itKey.first.theCharge!=charge) continue; //charge==0 means ignore charge
    AlgoMuon val = sortSingleResult(itKey.second);
    ///Accept only candidates with >2 hits
    if(val.getQ() < 3) continue;

    if(val.getQ() > (int)nHitsMax){
      nHitsMax = val.getQ();
      pdfValMax = val.getDisc();
      refPhi = val.getPhi();
      refEta = val.getEta();
      refLayer = val.getRefLayer();
      hitsWord = val.getHits();
      refPhiRHit = val.getPhiRHit();
      bestKey = itKey.first;
//      std::cout <<" sorter, byQual, now best is: "<<bestKey << std::endl;
    }
    else if(val.getQ() == (int)nHitsMax && val.getDisc() > (int)pdfValMax){
      pdfValMax = val.getDisc();
      refPhi = val.getPhi();
      refEta = val.getEta();
      refLayer = val.getRefLayer();
      hitsWord = val.getHits();
      refPhiRHit = val.getPhiRHit(); 
      bestKey = itKey.first;
//      std::cout <<" sorter, byDisc, now best is: "<<bestKey << std::endl;
    }
    else if(val.getQ() == (int)nHitsMax && val.getDisc() == (int)pdfValMax && itKey.first.number() < bestKey.number()) { 
//      itKey.first.thePtCode < bestKey.thePtCode){
      pdfValMax = val.getDisc();
      refPhi = val.getPhi();
      refEta = val.getEta();
      refLayer = val.getRefLayer();
      hitsWord = val.getHits();
      refPhiRHit = val.getPhiRHit(); 
      bestKey = itKey.first;
//      std::cout <<" sorter, byNumb, now best is: "<<bestKey << std::endl;
    }
  }

  AlgoMuon candidate(pdfValMax, refPhi, refEta, refLayer, 
                        hitsWord, nHitsMax, 0, 
                        bestKey.thePtCode, bestKey.theCharge);

  candidate.setPhiRHit(refPhiRHit); // for backward compatibility
  candidate.setPatternNumber(bestKey.number());

//  std::cout <<" return: " << candidate << std::endl; 
  return candidate;
}
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
void OMTFSorter::sortRefHitResults(const std::vector<OMTFProcessor::resultsMap> & procResults,
              std::vector<AlgoMuon> & refHitCands,
              int charge){
  
//  for(auto itRefHit: procResults) refHitCands.push_back(sortRefHitResults(itRefHit,charge));
  for (unsigned int iRefHit = 0 ; iRefHit < procResults.size(); iRefHit++) {
    AlgoMuon mu = sortRefHitResults( procResults[iRefHit],charge);
    mu.setRefHitNumber(iRefHit);
    refHitCands.push_back(mu);
  }
}
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
bool OMTFSorter::checkHitPatternValidity(unsigned int hits){

  ///FIXME: read the list from configuration so this can be controlled at runtime.
  std::vector<unsigned int> badPatterns = {99840, 34304, 3075, 36928, 12300, 98816, 98944, 33408, 66688, 66176, 7171, 20528, 33856, 35840, 4156, 34880};

  for(auto aHitPattern: badPatterns){
    if(hits==aHitPattern) return false;
  }

  return true; 
}
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
std::vector<l1t::RegionalMuonCand> OMTFSorter::candidates(unsigned int iProcessor, l1t::tftype mtfType, const std::vector<AlgoMuon> & algoCands)
{

  std::vector<l1t::RegionalMuonCand> result;

  for(auto myCand: algoCands){
    l1t::RegionalMuonCand candidate;
    candidate.setHwPt(myCand.getPt());
    candidate.setHwEta(myCand.getEta());

    int phiValue = myCand.getPhi();
    if(phiValue>= int(nPhiBins) ) phiValue-=nPhiBins;
    ///conversion factor from OMTF to uGMT scale is  5400/576 i.e. phiValue/=9.375;
    phiValue = floor(phiValue*437./pow(2,12));    // ie. use as in hw: 9.3729977 
    candidate.setHwPhi(phiValue);
    
    candidate.setHwSign(myCand.getCharge()<0 ? 1:0  );
    candidate.setHwSignValid(1);
 
    unsigned int quality = checkHitPatternValidity(myCand.getHits()) ? 0 | (1 << 2) | (1 << 3) 
                                                                     : 0 | (1 << 2);
    if (    abs(myCand.getEta()) == 115
        && (    static_cast<unsigned int>(myCand.getHits()) == std::bitset<18>("100000001110000000").to_ulong() 
             || static_cast<unsigned int>(myCand.getHits()) == std::bitset<18>("000000001110000000").to_ulong()
             || static_cast<unsigned int>(myCand.getHits()) == std::bitset<18>("100000000110000000").to_ulong()
             || static_cast<unsigned int>(myCand.getHits()) == std::bitset<18>("100000001100000000").to_ulong()
             || static_cast<unsigned int>(myCand.getHits()) == std::bitset<18>("100000001010000000").to_ulong()
           )
       ) quality = 4;
    if( myOmtfConfig->fwVersion() >= 5 ) {
      if (    static_cast<unsigned int>(myCand.getHits()) == std::bitset<18>("000000010000000011").to_ulong() 
           || static_cast<unsigned int>(myCand.getHits()) == std::bitset<18>("000000100000000011").to_ulong()
           || static_cast<unsigned int>(myCand.getHits()) == std::bitset<18>("000001000000000011").to_ulong()
           || static_cast<unsigned int>(myCand.getHits()) == std::bitset<18>("000010000000000011").to_ulong()
           || static_cast<unsigned int>(myCand.getHits()) == std::bitset<18>("000100000000000011").to_ulong()
           || static_cast<unsigned int>(myCand.getHits()) == std::bitset<18>("001000000000000011").to_ulong()
           || static_cast<unsigned int>(myCand.getHits()) == std::bitset<18>("010000000000000011").to_ulong()
           || static_cast<unsigned int>(myCand.getHits()) == std::bitset<18>("100000000000000011").to_ulong()
           || static_cast<unsigned int>(myCand.getHits()) == std::bitset<18>("000000010000001100").to_ulong() 
           || static_cast<unsigned int>(myCand.getHits()) == std::bitset<18>("000000100000001100").to_ulong()
           || static_cast<unsigned int>(myCand.getHits()) == std::bitset<18>("000001000000001100").to_ulong()
           || static_cast<unsigned int>(myCand.getHits()) == std::bitset<18>("000010000000001100").to_ulong()
           || static_cast<unsigned int>(myCand.getHits()) == std::bitset<18>("000100000000001100").to_ulong()
           || static_cast<unsigned int>(myCand.getHits()) == std::bitset<18>("001000000000001100").to_ulong()
           || static_cast<unsigned int>(myCand.getHits()) == std::bitset<18>("010000000000001100").to_ulong()
           || static_cast<unsigned int>(myCand.getHits()) == std::bitset<18>("100000000000001100").to_ulong()
           || static_cast<unsigned int>(myCand.getHits()) == std::bitset<18>("000000010000110000").to_ulong() 
           || static_cast<unsigned int>(myCand.getHits()) == std::bitset<18>("000000100000110000").to_ulong()
           || static_cast<unsigned int>(myCand.getHits()) == std::bitset<18>("000001000000110000").to_ulong()
           || static_cast<unsigned int>(myCand.getHits()) == std::bitset<18>("000010000000110000").to_ulong()
           || static_cast<unsigned int>(myCand.getHits()) == std::bitset<18>("000100000000110000").to_ulong()
           || static_cast<unsigned int>(myCand.getHits()) == std::bitset<18>("001000000000110000").to_ulong()
           || static_cast<unsigned int>(myCand.getHits()) == std::bitset<18>("010000000000110000").to_ulong()
           || static_cast<unsigned int>(myCand.getHits()) == std::bitset<18>("100000000000110000").to_ulong()
         ) quality = 1;
    }

//  if (abs(myCand.getEta()) == 121) quality = 4;
    if (abs(myCand.getEta()) == 121) quality = 0; // changed on request from HI

    candidate.setHwQual (quality);

    std::map<int, int> trackAddr;
    trackAddr[0] = myCand.getHits();
    trackAddr[1] = myCand.getRefLayer();
    trackAddr[2] = myCand.getDisc();
    candidate.setTrackAddress(trackAddr);
    candidate.setTFIdentifiers(iProcessor,mtfType);
    if (candidate.hwPt())  result.push_back(candidate); 
  }
  return result;
}
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
