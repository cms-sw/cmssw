/*
 * OMTFProcessor.cpp
 *
 *  Created on: Oct 7, 2017
 *      Author: kbunkow
 */

#include <L1Trigger/L1TMuonBayes/interface/Omtf/GhostBusterPreferRefDt.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/GoldenPatternWithStat.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/OMTFProcessor.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/OMTFSorterWithThreshold.h>
#include <iostream>
#include <algorithm>
#include <strstream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"



///////////////////////////////////////////////
///////////////////////////////////////////////
template <class GoldenPatternType>
OMTFProcessor<GoldenPatternType>::OMTFProcessor(OMTFConfiguration* omtfConfig, const edm::ParameterSet& edmCfg, edm::EventSetup const& evSetup, const L1TMuonOverlapParams* omtfPatterns, MuStubsInputTokens& muStubsInputTokens): ProcessorBase<GoldenPatternType>(omtfConfig, omtfPatterns)  {
  init(edmCfg, evSetup, muStubsInputTokens);
};


template <class GoldenPatternType>
OMTFProcessor<GoldenPatternType>::OMTFProcessor(OMTFConfiguration* omtfConfig, const edm::ParameterSet& edmCfg, edm::EventSetup const& evSetup, const typename ProcessorBase<GoldenPatternType>::GoldenPatternVec& gps, MuStubsInputTokens& muStubsInputTokens):
    ProcessorBase<GoldenPatternType>(omtfConfig, gps)
{
  init(edmCfg, evSetup, muStubsInputTokens);
};

template <class GoldenPatternType>
OMTFProcessor<GoldenPatternType>::~OMTFProcessor() {

}

template <class GoldenPatternType>
void OMTFProcessor<GoldenPatternType>::init(const edm::ParameterSet& edmCfg, edm::EventSetup const& evSetup, MuStubsInputTokens& muStubsInputTokens) {
  //TODO make it working....
/*  if(edmCfg.exists("sorterType") ) {//TODO add it also for the patternType == "GoldenPattern" - if needed
    string sorterType = edmCfg.getParameter<std::string>("sorterType");
    edm::LogImportant("OMTFReconstruction") << "OMTFProcessor constructed. sorterType: "<<sorterType<< std::endl;
    if(sorterType == "sorterWithThreshold") {
      GoldenPatternResult::setFinalizeFunction(2);

      typename OMTFSorterWithThreshold<GoldenPatternType>::Mode mode = OMTFSorterWithThreshold<GoldenPatternType>::bestGPByMaxGpProbability1;
      string modeStr = edmCfg.getParameter<std::string>("sorterWithThresholdMode");
      if(modeStr == "bestGPByThresholdOnProbability2")
        mode = OMTFSorterWithThreshold<GoldenPatternType>::bestGPByThresholdOnProbability2;
      else if(modeStr == "bestGPByMaxGpProbability1")
        mode = OMTFSorterWithThreshold<GoldenPatternType>::bestGPByMaxGpProbability1;

      setSorter(new OMTFSorterWithThreshold<GoldenPatternType>(this->myOmtfConfig, mode));
    }
  }
  else*/
    setSorter(new OMTFSorter<GoldenPatternType>()); //initialize with the default sorter

  if(edmCfg.exists("ghostBusterType") ) {
    if(edmCfg.getParameter<std::string>("ghostBusterType") == "GhostBusterPreferRefDt")
      setGhostBuster(new GhostBusterPreferRefDt(this->myOmtfConfig));
  }
  else if (this->myOmtfConfig->fwVersion() >= 5) {
    setGhostBuster(new GhostBusterPreferRefDt(this->myOmtfConfig) );
  }
  else {
    setGhostBuster(new GhostBuster()); //initialize with the default sorter
  }

  inputMaker.initialize(edmCfg, evSetup, this->myOmtfConfig, muStubsInputTokens);
}

template <class GoldenPatternType>
void OMTFProcessor<GoldenPatternType>::loadAndFilterDigis(const edm::Event& event, const edm::ParameterSet& edmCfg){
  inputMaker.loadAndFilterDigis(event);
}

template <class GoldenPatternType>
std::vector<l1t::RegionalMuonCand> OMTFProcessor<GoldenPatternType>::getFinalcandidates(unsigned int iProcessor, l1t::tftype mtfType, const AlgoMuons& algoCands) {

  std::vector<l1t::RegionalMuonCand> result;

  for(auto& myCand: algoCands) {
    l1t::RegionalMuonCand candidate;
    candidate.setHwPt(myCand->getPt());
    candidate.setHwEta(myCand->getEtaHw());

    int phiValue = myCand->getPhi();
    if(phiValue>= int(this->myOmtfConfig->nPhiBins()) )
      phiValue -= this->myOmtfConfig->nPhiBins();
    ///conversion factor from OMTF to uGMT scale is  5400/576 i.e. phiValue/=9.375;
    phiValue = floor(phiValue*437./pow(2,12));    // ie. use as in hw: 9.3729977
    candidate.setHwPhi(phiValue);

    candidate.setHwSign(myCand->getCharge()<0 ? 1:0  );
    candidate.setHwSignValid(1);

    unsigned int quality = checkHitPatternValidity(myCand->getFiredLayerBits()) ? 0 | (1 << 2) | (1 << 3)
                                                                     : 0 | (1 << 2);
    if (    abs(myCand->getEtaHw()) == 115
        && (    static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("100000001110000000").to_ulong()
             || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000001110000000").to_ulong()
             || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("100000000110000000").to_ulong()
             || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("100000001100000000").to_ulong()
             || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("100000001010000000").to_ulong()
           )
       ) quality =4;
    if( this->myOmtfConfig->fwVersion() >= 5 ) {
      if (    static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000010000000011").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000100000000011").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000001000000000011").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000010000000000011").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000100000000000011").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("001000000000000011").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("010000000000000011").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("100000000000000011").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000010000001100").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000100000001100").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000001000000001100").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000010000000001100").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000100000000001100").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("001000000000001100").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("010000000000001100").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("100000000000001100").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000010000110000").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000100000110000").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000001000000110000").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000010000000110000").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000100000000110000").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("001000000000110000").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("010000000000110000").to_ulong()
           || static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("100000000000110000").to_ulong()
         ) quality = 1;
    }
//  if (abs(myCand->getEta()) == 121) quality = 4;
    if (abs(myCand->getEtaHw()) == 121) quality = 0; // changed on request from HI

    candidate.setHwQual (quality);

    std::map<int, int> trackAddr;
    trackAddr[0] = myCand->getFiredLayerBits();
    trackAddr[1] = myCand->getRefLayer();
    trackAddr[2] = myCand->getDisc();
    candidate.setTrackAddress(trackAddr);
    candidate.setTFIdentifiers(iProcessor,mtfType);
    if (candidate.hwPt() > 0)  result.push_back(candidate);
  }
  return result;
}
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
template<class GoldenPatternType>
bool OMTFProcessor<GoldenPatternType>::checkHitPatternValidity(unsigned int hits) {
  ///FIXME: read the list from configuration so this can be controlled at runtime.
  std::vector<unsigned int> badPatterns = {99840, 34304, 3075, 36928, 12300, 98816, 98944, 33408, 66688, 66176, 7171, 20528, 33856, 35840, 4156, 34880};

  /*
99840 01100001 1000 000000
34304 00100001 1000 000000
 3075 00000011 0000 000011
36928 00100100 0001 000000
12300 00001100 0000 001100
98816 01100000 1000 000000
98944 01100000 1010 000000
33408 00100000 1010 000000
66688 01000001 0010 000000
66176 01000000 1010 000000
 7171 00000111 0000 000011
20528 00010100 0000 110000
33856 00100001 0001 000000
35840 00100011 0000 000000
 4156 00000100 0000 111100
34880 00100010 0001 000000
   */
  for(auto aHitPattern: badPatterns){
    if(hits==aHitPattern) return false;
  }

  return true;
}
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
template<class GoldenPatternType>
AlgoMuons OMTFProcessor<GoldenPatternType>::sortResults(unsigned int iProcessor, l1t::tftype mtfType, int charge) {
  unsigned int procIndx = this->myOmtfConfig->getProcIndx(iProcessor, mtfType);
  return sorter->sortResults(procIndx, this->getPatterns(), charge);
}
///////////////////////////////////////////////
///////////////////////////////////////////////
//const std::vector<OMTFProcessor::resultsMap> &
template<class GoldenPatternType>
const void OMTFProcessor<GoldenPatternType>::processInput(unsigned int iProcessor, l1t::tftype mtfType, const OMTFinput & aInput) {
  unsigned int procIndx = this->myOmtfConfig->getProcIndx(iProcessor, mtfType);
  for(auto& itGP: this->theGPs) {
    for(auto& result : itGP->getResults()[procIndx]) {
      result.reset();
    }
  }

  //////////////////////////////////////
  //////////////////////////////////////
  std::bitset<128> refHitsBits = aInput.getRefHits(iProcessor);
  if(refHitsBits.none())
    return; // myResults;

  for(unsigned int iLayer=0; iLayer < this->myOmtfConfig->nLayers(); ++iLayer) {
    /*for(auto& h : layerHits) {
      if(h != 5400)
        std::cout<<__FUNCTION__<<" "<<__LINE__<<" iLayer "<<iLayer<<" layerHit "<<h<<std::endl;
    }*/
    ///Number of reference hits to be checked.
    unsigned int nTestedRefHits = this->myOmtfConfig->nTestRefHits();
    for(unsigned int iRefHit = 0; iRefHit < this->myOmtfConfig->nRefHits(); ++iRefHit) { //loop over all possible refHits, i.e. 128
      if(!refHitsBits[iRefHit]) continue;
      if(nTestedRefHits-- == 0) break;

      const RefHitDef& aRefHitDef = this->myOmtfConfig->getRefHitsDefs()[iProcessor][iRefHit];


      unsigned int refLayerLogicNum = this->myOmtfConfig->getRefToLogicNumber()[aRefHitDef.iRefLayer];
      const MuonStubPtr refStub = aInput.getMuonStub(refLayerLogicNum, aRefHitDef.iInput);
      int phiRef = refStub->phiHw;
      int etaRef = refStub->etaHw;

      unsigned int iRegion = aRefHitDef.iRegion;

      if(this->myOmtfConfig->getBendingLayers().count(iLayer)) //this iLayer is a banding layer
        phiRef = 0;  //then in the delta_phi in process1Layer1RefLayer one obtains simply the iLayer phi

      MuonStubPtrs1D restrictedLayerStubs = this->restrictInput(iProcessor, iRegion, iLayer, aInput);

      //std::cout<<__FUNCTION__<<" "<<__LINE__<<" iLayer "<<iLayer<<" iRefLayer "<<aRefHitDef.iRefLayer<<" hits.size "<<restrictedLayerHits.size()<<std::endl;
      //std::cout<<"iLayer "<<iLayer<<" refHitNum "<<myOmtfConfig->nTestRefHits()-nTestedRefHits-1<<" iRefHit "<<iRefHit;
      //std::cout<<" nTestedRefHits "<<nTestedRefHits<<" aRefHitDef "<<aRefHitDef<<std::endl;

      int refLayerLogicNumber = this->myOmtfConfig->getRefToLogicNumber()[aRefHitDef.iRefLayer];

      unsigned int refHitNumber = this->myOmtfConfig->nTestRefHits()-nTestedRefHits-1;
      for(auto& itGP: this->theGPs) {
        if(itGP->key().thePt == 0) //empty pattern
          continue;

        StubResult stubResult = itGP->process1Layer1RefLayer(aRefHitDef.iRefLayer, iLayer,
            restrictedLayerStubs,
            refStub);

        int phiRefSt2 = itGP->propagateRefPhi(phiRef, etaRef, aRefHitDef.iRefLayer); //fixme this unnecessary repeated  for every layer

        //std::cout<<__FUNCTION__<<":"<<__LINE__<<" layerResult: valid"<<layerResult.valid<<" pdfVal "<<layerResult.pdfVal<<std::endl;
        itGP->getResults()[procIndx][refHitNumber].setStubResult(iLayer, stubResult);
        itGP->getResults()[procIndx][refHitNumber].set(aRefHitDef.iRefLayer, phiRefSt2, etaRef, phiRef); //fixme this unnecessary repeated  for every layer
      }
    }
  }
  //////////////////////////////////////
  //////////////////////////////////////
  {
    for(auto& itGP: this->theGPs) {
      itGP->finalise(procIndx);
      //debug
      /*for(unsigned int iRefHit = 0; iRefHit < itGP->getResults()[procIndx].size(); ++iRefHit) {
        if(itGP->getResults()[procIndx][iRefHit].isValid()) {
          std::cout<<__FUNCTION__<<":"<<"__LINE__"<<itGP->getResults()[procIndx][iRefHit]<<std::endl;
        }
      }*/
    }
  }

/*  std::ostringstream myStr;
  myStr<<"iProcessor: "<<iProcessor<<std::endl;
  myStr<<"Input: ------------"<<std::endl;
  myStr<<aInput<<std::endl;
  edm::LogInfo("OMTF processor")<<myStr.str();*/

  return;
}
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////

template<class GoldenPatternType>
std::vector<l1t::RegionalMuonCand> OMTFProcessor<GoldenPatternType>::
run(unsigned int iProcessor, l1t::tftype mtfType, int bx, std::vector<std::unique_ptr<IOMTFEmulationObserver> >& observers) {
  //boost::timer::auto_cpu_timer t("%ws wall, %us user in getProcessorCandidates\n");
  inputMaker.setFlag(0);

  OMTFinput input(this->myOmtfConfig);
  inputMaker.buildInputForProcessor(input.getMuonStubs(),
                iProcessor, mtfType, bx, bx);
  int flag = inputMaker.getFlag();

  //cout<<"buildInputForProce "; t.report();
  processInput(iProcessor, mtfType, input);

  //cout<<"processInput       "; t.report();
  AlgoMuons algoCandidates =  sortResults(iProcessor, mtfType);

  //cout<<"sortResults        "; t.report();
  // perform GB
  AlgoMuons gbCandidates =  ghostBust(algoCandidates);

  //cout<<"ghostBust"; t.report();
  // fill RegionalMuonCand colleciton
  std::vector<l1t::RegionalMuonCand> candMuons = getFinalcandidates(iProcessor, mtfType, gbCandidates);

  //cout<<"getFinalcandidates "; t.report();
  //fill outgoing collection
  for (auto & candMuon :  candMuons) {
     candMuon.setHwQual( candMuon.hwQual() | flag);         //FIXME temporary debug fix
  }

  //dump to XML
  //if(bx==0) writeResultToXML(iProcessor, mtfType,  input, algoCandidates, candMuons); //TODO handle bx
  //if(bx==0)
  for(auto& obs : observers) {
    obs->observeProcesorEmulation(iProcessor, mtfType,  input, algoCandidates, gbCandidates, candMuons);
  }

  return candMuons;
}

/////////////////////////////////////////////////////////

template class OMTFProcessor<GoldenPattern>;
template class OMTFProcessor<GoldenPatternWithStat>;
template class OMTFProcessor<GoldenPatternWithThresh>;
