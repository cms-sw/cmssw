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
    }
    else if(val.getQ() == (int)nHitsMax && val.getDisc() > (int)pdfValMax){
      pdfValMax = val.getDisc();
      refPhi = val.getPhi();
      refEta = val.getEta();
      refLayer = val.getRefLayer();
      hitsWord = val.getHits();
      refPhiRHit = val.getPhiRHit(); 
      bestKey = itKey.first;
    }
    else if(val.getQ() == (int)nHitsMax && val.getDisc() == (int)pdfValMax &&
      itKey.first.thePtCode < bestKey.thePtCode){
      pdfValMax = val.getDisc();
      refPhi = val.getPhi();
      refEta = val.getEta();
      refLayer = val.getRefLayer();
      hitsWord = val.getHits();
      refPhiRHit = val.getPhiRHit(); 
      bestKey = itKey.first;
    }
  }

  AlgoMuon candidate(pdfValMax, refPhi, refEta, refLayer, 
                        hitsWord, nHitsMax, 0, 
                        bestKey.thePtCode, bestKey.theCharge);

  candidate.setPhiRHit(refPhiRHit); // for backward compatibility

  return candidate;
}
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
void OMTFSorter::sortRefHitResults(const std::vector<OMTFProcessor::resultsMap> & procResults,
              std::vector<AlgoMuon> & refHitCands,
              int charge){
  
  for(auto itRefHit: procResults) refHitCands.push_back(sortRefHitResults(itRefHit,charge));
}
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// AlgoMuon OMTFSorter::sortProcessorResults(const std::vector<OMTFProcessor::resultsMap> & procResults,
// 					     int charge){ //method kept for backward compatibility

//   std::vector<AlgoMuon> sortedCandidates;
//   sortProcessorResults(procResults, sortedCandidates, charge);

//   AlgoMuon candidate = sortedCandidates.size()>0 ? sortedCandidates[0] : AlgoMuon(0,999,9999,0,0,0,0,-1);

//   std::ostringstream myStr;
//   myStr<<"Selected Candidate with charge: "<<charge<<" "<<candidate<<std::endl;
//   edm::LogInfo("OMTF Sorter")<<myStr.str();

//   return candidate;

// }
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// void OMTFSorter::sortProcessorResults(const std::vector<OMTFProcessor::resultsMap> & procResults,
//               std::vector<AlgoMuon> & refHitCleanCands,
//               int charge){

//   refHitCleanCands.clear();
//   std::vector<AlgoMuon> refHitCands;

//   for(auto itRefHit: procResults) refHitCands.push_back(sortRefHitResults(itRefHit,charge));

//   // Sort candidates with decreased goodness,
//   // where goodness definied in < operator of AlgoMuon
//   std::sort( refHitCands.begin(), refHitCands.end() );

//   // Clean candidate list by removing dupicates basing on Phi distance.
//   // Assumed that the list is ordered
//   for(std::vector<AlgoMuon>::iterator it1 = refHitCands.begin();
//       it1 != refHitCands.end(); ++it1){
//     bool isGhost=false;
//     for(std::vector<AlgoMuon>::iterator it2 = refHitCleanCands.begin();
//   it2 != refHitCleanCands.end(); ++it2){
//       //do not accept candidates with similar phi (any charge combination)
//       //veto window 5deg(=half of logic cone)=5/360*5760=80"logic strips"
//       if(std::abs(it1->phi - it2->phi)<5/360.0*OMTFConfiguration::instance()->nPhiBins){
//   isGhost=true;
//   break;
//       }
//     }
//     if(it1->q>0 && !isGhost) refHitCleanCands.push_back(*it1);
//   }
//   //return 3 candidates (adding empty ones if needed)
//   refHitCleanCands.resize( 3, AlgoMuon(0,999,9999,0,0,0,0,0) );

//   std::ostringstream myStr;
//   bool hasCandidates = false;
//   for(unsigned int iRefHit=0;iRefHit<refHitCands.size();++iRefHit){
//     if(refHitCands[iRefHit].q){
//       hasCandidates=true;
//       break;
//     }
//   }
//   for(unsigned int iRefHit=0;iRefHit<refHitCands.size();++iRefHit){
//     if(refHitCands[iRefHit].q) myStr<<"Ref hit: "<<iRefHit<<" "<<refHitCands[iRefHit]<<std::endl;
//   }
//   myStr<<"Selected Candidates with charge: "<<charge<<std::endl;
//   for(unsigned int iCand=0; iCand<refHitCleanCands.size(); ++iCand){
//     myStr<<"Cand: "<<iCand<<" "<<refHitCleanCands[iCand]<<std::endl;
//   }

//   if(hasCandidates) edm::LogInfo("OMTF Sorter")<<myStr.str();

//   return;
// }

///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// l1t::RegionalMuonCand OMTFSorter::sortProcessor(const std::vector<OMTFProcessor::resultsMap> & procResults,
// 							int charge){ //method kept for backward compatibility

//   AlgoMuon myCand = sortProcessorResults(procResults, charge);

//   l1t::RegionalMuonCand candidate;
//   std::bitset<17> bits(myCand.getHits());
//   int ipt = myCand.getPt()+1;
//   if(ipt>31) ipt=31;
//   candidate.setHwPt(RPCConst::ptFromIpt(ipt)*2.0);//MicroGMT has 0.5 GeV pt bins
//   candidate.setHwEta(myCand.getEta());//eta scale set during input making in OMTFInputmaker
//   candidate.setHwPhi(myCand.getPhi());
//   candidate.setHwSign(myCand.getCharge()+1*(myCand.getCharge()<0));
//   candidate.setHwQual(bits.count());
//   std::map<int, int> trackAddr;
//   trackAddr[0] = myCand.getHits();
//   trackAddr[1] = myCand.getRefLayer();
//   trackAddr[2] = myCand.getDisc();
//   candidate.setTrackAddress(trackAddr);
  
//   return candidate;
// }
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
void OMTFSorter::sortProcessorAndFillCandidates(unsigned int iProcessor, l1t::tftype mtfType,
                 const std::vector<AlgoMuon> & algoCands,
                 l1t::RegionalMuonCandBxCollection & sortedCands,
                 int bx, int charge){

  for(auto myCand: algoCands){
    l1t::RegionalMuonCand candidate;
    std::bitset<17> bits(myCand.getHits());
    candidate.setHwPt(myCand.getPt());
    candidate.setHwEta(myCand.getEta());

    float phiValue = myCand.getPhi();
    if(phiValue>=(int)OMTFConfiguration::instance()->nPhiBins) phiValue-=OMTFConfiguration::instance()->nPhiBins;
    ///conversion factor from OMTF to uGMT scale: 5400/576
    phiValue/=9.375;
    candidate.setHwPhi(phiValue);
    
    candidate.setHwSign(myCand.getCharge()<0 ? 1:0  );
    //FIXME: Obsolete 
    ///Quality is set to number of leayers hit.
    ///DT bending and position hit is counted as one.
    ///thus we subtract 1 for each DT station hit.
    //candidate.setHwQual(bits.count() - bits.test(0) - bits.test(2) - bits.test(4));
    ///Candidates with bad hit patterns get quality 0.
    //if(!checkHitPatternValidity(myCand.getHits())) candidate.setHwQual(0);

    /// Now quality based on hit pattern  
    /// 
    unsigned int quality = checkHitPatternValidity(myCand.getHits()) ? 0 | (1 << 2) | (1 << 3) 
                                                                     : 0 | (1 << 2);
    candidate.setHwQual ( quality);

    std::map<int, int> trackAddr;
    trackAddr[0] = myCand.getHits();
    trackAddr[1] = myCand.getRefLayer();
    trackAddr[2] = myCand.getDisc();
    candidate.setTrackAddress(trackAddr);
    candidate.setTFIdentifiers(iProcessor,mtfType);
    if(candidate.hwPt()) sortedCands.push_back(bx, candidate);
  }
}
