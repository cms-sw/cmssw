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
std::tuple<unsigned int,unsigned int, int, int, unsigned int, int> OMTFSorter::sortSingleResult(const OMTFResult & aResult){

  OMTFResult::vector1D pdfValsVec = aResult.getSummaryVals();
  OMTFResult::vector1D nHitsVec = aResult.getSummaryHits();
  OMTFResult::vector1D refPhiVec = aResult.getRefPhis();
  OMTFResult::vector1D refEtaVec = aResult.getRefEtas();
  OMTFResult::vector1D hitsVec = aResult.getHitsWord();

  assert(pdfValsVec.size()==nHitsVec.size());

  unsigned int nHitsMax = 0;
  unsigned int pdfValMax = 0;
  unsigned int hitsWord = 0;
  int refPhi = 1024;
  int refEta = -10;
  int refLayer = -1;

  std::tuple<unsigned int,unsigned int, int, int, unsigned int, int>  sortedResult;
  std::get<0>(sortedResult) = nHitsMax;
  std::get<1>(sortedResult) = pdfValMax;
  std::get<2>(sortedResult) = refPhi;
  std::get<3>(sortedResult) = refLayer;
  std::get<4>(sortedResult) = hitsWord;
  std::get<5>(sortedResult) = refEta;

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
      }
    }
  }

  std::get<0>(sortedResult) = nHitsMax;
  std::get<1>(sortedResult) = pdfValMax;
  std::get<2>(sortedResult) = refPhi;
  std::get<3>(sortedResult) = refLayer;
  std::get<4>(sortedResult) = hitsWord;
  std::get<5>(sortedResult) = refEta;
  return sortedResult;
}
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
InternalObj OMTFSorter::sortRefHitResults(const OMTFProcessor::resultsMap & aResultsMap,
					  int charge){

  unsigned int pdfValMax = 0;
  unsigned int nHitsMax = 0;
  unsigned int hitsWord = 0;
  int refPhi = 9999;
  int refEta = 999;
  int refLayer = -1;
  Key bestKey;
  for(auto itKey: aResultsMap){
    if(charge!=0 && itKey.first.theCharge!=charge) continue; //charge==0 means ignore charge
    std::tuple<unsigned int,unsigned int, int, int, unsigned int, int > val = sortSingleResult(itKey.second);
    ///Accept only candidates with >2 hits
    if(std::get<0>(val)<3) continue;
    if( std::get<0>(val)>nHitsMax){
      nHitsMax = std::get<0>(val);
      pdfValMax = std::get<1>(val);
      refPhi = std::get<2>(val);
      refEta = std::get<5>(val);
      refLayer = std::get<3>(val);
      hitsWord = std::get<4>(val);
      bestKey = itKey.first;
    }
    else if(std::get<0>(val)==nHitsMax && std::get<1>(val)>pdfValMax){
      pdfValMax = std::get<1>(val);
      refPhi = std::get<2>(val);
      refEta = std::get<5>(val);
      refLayer = std::get<3>(val);
      hitsWord = std::get<4>(val);
      bestKey = itKey.first;
    }
    else if(std::get<0>(val)==nHitsMax && std::get<1>(val)==pdfValMax &&
	    itKey.first.thePtCode<bestKey.thePtCode){
      pdfValMax = std::get<1>(val);
      refPhi = std::get<2>(val);
      refEta = std::get<5>(val);
      refLayer = std::get<3>(val);
      hitsWord = std::get<4>(val);
      bestKey = itKey.first;
    }
  }

  InternalObj candidate(bestKey.thePtCode, refEta, refPhi,
			pdfValMax, 0, nHitsMax,
			bestKey.theCharge, refLayer);

  candidate.hits   = hitsWord;

  return candidate;
}
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
InternalObj OMTFSorter::sortProcessorResults(const std::vector<OMTFProcessor::resultsMap> & procResults,
					     int charge){ //method kept for backward compatibility

  std::vector<InternalObj> sortedCandidates;
  sortProcessorResults(procResults, sortedCandidates, charge);

  InternalObj candidate = sortedCandidates.size()>0 ? sortedCandidates[0] : InternalObj(0,999,9999,0,0,0,0,-1);

  std::ostringstream myStr;
  myStr<<"Selected Candidate with charge: "<<charge<<" "<<candidate<<std::endl;
  edm::LogInfo("OMTF Sorter")<<myStr.str();

  return candidate;

}
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
void OMTFSorter::sortProcessorResults(const std::vector<OMTFProcessor::resultsMap> & procResults,
				      std::vector<InternalObj> & refHitCleanCands,
				      int charge){

  refHitCleanCands.clear();
  std::vector<InternalObj> refHitCands;

  for(auto itRefHit: procResults) refHitCands.push_back(sortRefHitResults(itRefHit,charge));

  // Sort candidates with decreased goodness,
  // where goodness definied in < operator of InternalObj
  std::sort( refHitCands.begin(), refHitCands.end() );

  // Clean candidate list by removing dupicates basing on Phi distance.
  // Assumed that the list is ordered
  for(std::vector<InternalObj>::iterator it1 = refHitCands.begin();
      it1 != refHitCands.end(); ++it1){
    bool isGhost=false;
    for(std::vector<InternalObj>::iterator it2 = refHitCleanCands.begin();
	it2 != refHitCleanCands.end(); ++it2){
      //do not accept candidates with similar phi (any charge combination)
      //veto window 5deg(=half of logic cone)=5/360*5760=80"logic strips"
      if(std::abs(it1->phi - it2->phi)<5/360.0*OMTFConfiguration::nPhiBins){
	isGhost=true;
	break;
      }
    }
    if(it1->q>0 && !isGhost) refHitCleanCands.push_back(*it1);
  }
  //return 3 candidates (adding empty ones if needed)
  refHitCleanCands.resize( 3, InternalObj(0,999,9999,0,0,0,0,0) );

  std::ostringstream myStr;
  bool hasCandidates = false;
  for(unsigned int iRefHit=0;iRefHit<refHitCands.size();++iRefHit){
    if(refHitCands[iRefHit].q){
      hasCandidates=true;
      break;
    }
  }
  for(unsigned int iRefHit=0;iRefHit<refHitCands.size();++iRefHit){
    if(refHitCands[iRefHit].q) myStr<<"Ref hit: "<<iRefHit<<" "<<refHitCands[iRefHit]<<std::endl;
  }
  myStr<<"Selected Candidates with charge: "<<charge<<std::endl;
  for(unsigned int iCand=0; iCand<refHitCleanCands.size(); ++iCand){
    myStr<<"Cand: "<<iCand<<" "<<refHitCleanCands[iCand]<<std::endl;
  }

  if(hasCandidates) edm::LogInfo("OMTF Sorter")<<myStr.str();

  return;
}
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
l1t::RegionalMuonCand OMTFSorter::sortProcessor(const std::vector<OMTFProcessor::resultsMap> & procResults,
							int charge){ //method kept for backward compatibility

  InternalObj myCand = sortProcessorResults(procResults, charge);

  l1t::RegionalMuonCand candidate;
  std::bitset<17> bits(myCand.hits);
  int ipt = myCand.pt+1;
  if(ipt>31) ipt=31;
  candidate.setHwPt(RPCConst::ptFromIpt(ipt)*2.0);//MicroGMT has 0.5 GeV pt bins
  candidate.setHwEta(myCand.eta);//eta scale set during input making in OMTFInputmaker
  candidate.setHwPhi(myCand.phi);
  candidate.setHwSign(myCand.charge+1*(myCand.charge<0));
  candidate.setHwQual(bits.count());
  std::map<int, int> trackAddr;
  trackAddr[0] = myCand.hits;
  trackAddr[1] = myCand.refLayer;
  trackAddr[2] = myCand.disc;
  candidate.setTrackAddress(trackAddr);
  
  return candidate;
}
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
void OMTFSorter::sortProcessor(const std::vector<OMTFProcessor::resultsMap> & procResults,
			       l1t::RegionalMuonCandBxCollection & sortedCands,
			       int bx, int charge){

  sortedCands.clear();
  std::vector<InternalObj> mySortedCands;
  sortProcessorResults(procResults, mySortedCands, charge);

  for(auto myCand: mySortedCands){
    l1t::RegionalMuonCand candidate;
    std::bitset<17> bits(myCand.hits);
    int ipt = myCand.pt+1;
    if(ipt>31) ipt=31;
    candidate.setHwPt(RPCConst::ptFromIpt(ipt)*2.0);//MicroGMT has 0.5 GeV pt bins
    candidate.setHwEta(myCand.eta);//eta scale set during input making in OMTFInputmaker
    candidate.setHwPhi(myCand.phi);
    candidate.setHwSign(myCand.charge+1*(myCand.charge<0));
    ///Quality is set to number of leayers hit.
    ///DT bending and position hit is counted as one.
    ///thus we subtract 1 for each DT station hit.
    candidate.setHwQual(bits.count() - bits.test(0) - bits.test(2) - bits.test(4));
    std::map<int, int> trackAddr;
    trackAddr[0] = myCand.hits;
    trackAddr[1] = myCand.refLayer;
    trackAddr[2] = myCand.disc;
    candidate.setTrackAddress(trackAddr);

    sortedCands.push_back(bx, candidate);
  }

  return;
}
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////

