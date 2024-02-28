/*
 * OMTFProcessor.cpp
 *
 *  Created on: Oct 7, 2017
 *      Author: kbunkow
 */
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFProcessor.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/MuonStub.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/MuonStubsInput.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GhostBuster.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GhostBusterPreferRefDt.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPatternWithStat.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/IOMTFEmulationObserver.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFinput.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFSorter.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/StubResult.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <bitset>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <map>
#include <string>
#include <vector>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

///////////////////////////////////////////////
///////////////////////////////////////////////
template <class GoldenPatternType>
OMTFProcessor<GoldenPatternType>::OMTFProcessor(OMTFConfiguration* omtfConfig,
                                                const edm::ParameterSet& edmCfg,
                                                edm::EventSetup const& evSetup,
                                                const L1TMuonOverlapParams* omtfPatterns)
    : ProcessorBase<GoldenPatternType>(omtfConfig, omtfPatterns) {
  init(edmCfg, evSetup);
};

template <class GoldenPatternType>
OMTFProcessor<GoldenPatternType>::OMTFProcessor(OMTFConfiguration* omtfConfig,
                                                const edm::ParameterSet& edmCfg,
                                                edm::EventSetup const& evSetup,
                                                GoldenPatternVec<GoldenPatternType>&& gps)
    : ProcessorBase<GoldenPatternType>(omtfConfig, std::forward<GoldenPatternVec<GoldenPatternType> >(gps)) {
  init(edmCfg, evSetup);
};

template <class GoldenPatternType>
OMTFProcessor<GoldenPatternType>::~OMTFProcessor() {
  if (useFloatingPointExtrapolation)
    saveExtrapolFactors();
}

template <class GoldenPatternType>
void OMTFProcessor<GoldenPatternType>::init(const edm::ParameterSet& edmCfg, edm::EventSetup const& evSetup) {
  setSorter(new OMTFSorter<GoldenPatternType>(this->myOmtfConfig->getSorterType()));
  //initialize with the default sorter

  if (this->myOmtfConfig->getGhostBusterType() == "GhostBusterPreferRefDt" ||
      this->myOmtfConfig->getGhostBusterType() == "byLLH" || this->myOmtfConfig->getGhostBusterType() == "byFPLLH" ||
      this->myOmtfConfig->getGhostBusterType() == "byRefLayer") {
    setGhostBuster(new GhostBusterPreferRefDt(this->myOmtfConfig));
    edm::LogVerbatim("OMTFReconstruction") << "setting " << this->myOmtfConfig->getGhostBusterType() << std::endl;
  } else {
    setGhostBuster(new GhostBuster(this->myOmtfConfig));  //initialize with the default sorter
    edm::LogVerbatim("OMTFReconstruction") << "setting GhostBuster" << std::endl;
  }

  edm::LogVerbatim("OMTFReconstruction") << "fwVersion 0x" << hex << this->myOmtfConfig->fwVersion() << std::endl;

  useStubQualInExtr = this->myOmtfConfig->useStubQualInExtr();
  useEndcapStubsRInExtr = this->myOmtfConfig->useEndcapStubsRInExtr();

  if (edmCfg.exists("useFloatingPointExtrapolation"))
    useFloatingPointExtrapolation = edmCfg.getParameter<bool>("useFloatingPointExtrapolation");

  std::string extrapolFactorsFilename;
  if (edmCfg.exists("extrapolFactorsFilename")) {
    extrapolFactorsFilename = edmCfg.getParameter<edm::FileInPath>("extrapolFactorsFilename").fullPath();
  }

  if (this->myOmtfConfig->usePhiBExtrapolationMB1() || this->myOmtfConfig->usePhiBExtrapolationMB2()) {
    extrapolFactors.resize(2, std::vector<std::map<int, double> >(this->myOmtfConfig->nLayers()));
    extrapolFactorsNorm.resize(2, std::vector<std::map<int, int> >(this->myOmtfConfig->nLayers()));

    //when useFloatingPointExtrapolation is true the extrapolFactors are not used,
    //all calculations are done in the extrapolateDtPhiBFloatPoint
    if (!extrapolFactorsFilename.empty() && !useFloatingPointExtrapolation)
      loadExtrapolFactors(extrapolFactorsFilename);
  }

  edm::LogVerbatim("OMTFReconstruction") << "useFloatingPointExtrapolation " << useFloatingPointExtrapolation
                                         << std::endl;
  edm::LogVerbatim("OMTFReconstruction") << "extrapolFactorsFilename " << extrapolFactorsFilename << std::endl;
}

template <class GoldenPatternType>
std::vector<l1t::RegionalMuonCand> OMTFProcessor<GoldenPatternType>::getFinalcandidates(unsigned int iProcessor,
                                                                                        l1t::tftype mtfType,
                                                                                        const AlgoMuons& algoCands) {
  std::vector<l1t::RegionalMuonCand> result;

  for (auto& myCand : algoCands) {
    l1t::RegionalMuonCand candidate;

    //the charge is only for the constrained measurement. The constrained measurement is always defined for a valid candidate
    if (ptAssignment) {
      candidate.setHwPt(myCand->getPtNNConstr());
      candidate.setHwSign(myCand->getChargeNNConstr() < 0 ? 1 : 0);
    } else {
      candidate.setHwPt(myCand->getPtConstr());
      candidate.setHwSign(myCand->getChargeConstr() < 0 ? 1 : 0);
    }

    if (mtfType == l1t::omtf_pos)
      candidate.setHwEta(myCand->getEtaHw());
    else
      candidate.setHwEta((-1) * myCand->getEtaHw());

    int phiValue = myCand->getPhi();
    if (phiValue >= int(this->myOmtfConfig->nPhiBins()))
      phiValue -= this->myOmtfConfig->nPhiBins();
    phiValue = this->myOmtfConfig->procPhiToGmtPhi(phiValue);
    candidate.setHwPhi(phiValue);

    candidate.setHwSignValid(1);

    if (myCand->getPtUnconstr() >= 0) {  //empty PtUnconstrained is -1, maybe should be corrected on the source
      //the upt has different hardware scale than the pt, the upt unit is 1 GeV
      candidate.setHwPtUnconstrained(myCand->getPtUnconstr());
    } else
      candidate.setHwPtUnconstrained(0);

    unsigned int quality = 12;
    if (this->myOmtfConfig->fwVersion() <= 6)
      quality = checkHitPatternValidity(myCand->getFiredLayerBits()) ? 0 | (1 << 2) | (1 << 3) : 0 | (1 << 2);  //12 : 4

    if (abs(myCand->getEtaHw()) == 115 &&  //115 is eta 1.25                         rrrrrrrrccccdddddd
        (static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("100000001110000000").to_ulong() ||
         static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000001110000000").to_ulong() ||
         static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("100000000110000000").to_ulong() ||
         static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("100000001100000000").to_ulong() ||
         static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("100000001010000000").to_ulong())) {
      if (this->myOmtfConfig->fwVersion() <= 6)
        quality = 4;
      else
        quality = 1;
    }

    if (this->myOmtfConfig->fwVersion() >= 5 && this->myOmtfConfig->fwVersion() <= 6) {
      if (static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000010000000011").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000100000000011").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000001000000000011").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000010000000000011").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000100000000000011").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("001000000000000011").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("010000000000000011").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("100000000000000011").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000010000001100").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000100000001100").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000001000000001100").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000010000000001100").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000100000000001100").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("001000000000001100").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("010000000000001100").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("100000000000001100").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000010000110000").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000100000110000").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000001000000110000").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000010000000110000").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000100000000110000").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("001000000000110000").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("010000000000110000").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("100000000000110000").to_ulong())
        quality = 1;
    } else if (this->myOmtfConfig->fwVersion() >= 8) {  //TODO fix the fwVersion     rrrrrrrrccccdddddd
      if (static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000110000000011").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000100000000011").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000010000000011").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000110000000001").to_ulong() ||

          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000001000000001100").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000011000000001100").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000010000000001100").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000011000000000100").to_ulong() ||

          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000011000000001").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("001000010000000001").to_ulong())
        quality = 1;
      else if (
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000010000000101").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000010001000001").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000011000000001").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000011000000011").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000011100000001").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000100000000011").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000100001000100").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000100100000001").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000110100000001").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000111000000000").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000111000000001").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000111000000011").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000001000001000100").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000001010000000001").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000001010000000011").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000001010000000100").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000001100000000001").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000001100000000100").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000001100000000111").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000001100001000000").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000001110000000100").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000001110000000101").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000010000000000101").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000010010000000001").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000010010000000100").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000010010000000101").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000010100000000001").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000010100000000101").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000011110000000100").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000011110000000101").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000101000000010101").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("001000010000000001").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("001000011000000000").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("001000011000000001").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("001000100000000001").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("001000110000000000").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("001001000000000100").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("001001100000000100").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("001010000000000100").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("010000000010000001").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("010000000011000100").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("010000010000000001").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("010000100000000001").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("100000011000000000").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000110000000001").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000010000000011").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000110000000011").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000011000000001100").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000011000000000100").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000010010000001").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000010000000001100").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("001001000001000100").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000001100000000101").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000100000000101").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000001100000000011").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000001110000000111").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("001000110001000001").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000001110000000011").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("001000000001000100").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000110001000001").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000001000000000101").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("001010000001000000").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("001100000001000000").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("100000010000000001").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("010000010010000000").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000010100000001100").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("001000110000000011").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000001000000001100").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000000000000111101").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000001100000110001").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000100000000010100").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("001000100000000011").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("001000110000000001").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("010000100010000001").to_ulong() ||
          static_cast<unsigned int>(myCand->getFiredLayerBits()) == std::bitset<18>("000100000000110000").to_ulong())
        quality = 8;
    }  //  if (abs(myCand->getEta()) == 121) quality = 4;
    if (abs(myCand->getEtaHw()) >= 121)
      quality = 0;  // changed from 4 on request from HI

    candidate.setHwQual(quality);

    std::map<int, int> trackAddr;
    trackAddr[0] = myCand->getFiredLayerBits();
    //TODO in the hardware, the uPt is sent to the uGMT at the trackAddr = (uPt << 18) + trackAddr;
    //check if it matters if it needs to be here as well
    trackAddr[1] = myCand->getRefLayer();
    trackAddr[2] = myCand->getDisc();
    trackAddr[3] = myCand->getGpResultUnconstr().getPdfSumUnconstr();
    if (candidate.hwPt() > 0) {
      candidate.setTrackAddress(trackAddr);
      candidate.setTFIdentifiers(iProcessor, mtfType);
      result.push_back(candidate);
    }
  }
  return result;
}
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
template <class GoldenPatternType>
bool OMTFProcessor<GoldenPatternType>::checkHitPatternValidity(unsigned int hits) {
  ///FIXME: read the list from configuration so this can be controlled at runtime.
  std::vector<unsigned int> badPatterns = {
      99840, 34304, 3075, 36928, 12300, 98816, 98944, 33408, 66688, 66176, 7171, 20528, 33856, 35840, 4156, 34880};

  /*
99840 01100001 1000 000000      011000011000000000
34304 00100001 1000 000000      001000011000000000
 3075 00000011 0000 000011      000000110000000011
36928 00100100 0001 000000      001001000001000000
12300 00001100 0000 001100      000011000000001100
98816 01100000 1000 000000      011000001000000000
98944 01100000 1010 000000      011000001010000000
33408 00100000 1010 000000      001000001010000000
66688 01000001 0010 000000      010000010010000000
66176 01000000 1010 000000      010000001010000000
 7171 00000111 0000 000011      000001110000000011
20528 00010100 0000 110000      000101000000110000
33856 00100001 0001 000000      001000010001000000
35840 00100011 0000 000000      001000110000000000
 4156 00000100 0000 111100      000001000000111100
34880 00100010 0001 000000      001000100001000000
   */
  for (auto aHitPattern : badPatterns) {
    if (hits == aHitPattern)
      return false;
  }

  return true;
}
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
template <class GoldenPatternType>
AlgoMuons OMTFProcessor<GoldenPatternType>::sortResults(unsigned int iProcessor, l1t::tftype mtfType, int charge) {
  unsigned int procIndx = this->myOmtfConfig->getProcIndx(iProcessor, mtfType);
  return sorter->sortResults(procIndx, this->getPatterns(), charge);
}

template <class GoldenPatternType>
int OMTFProcessor<GoldenPatternType>::extrapolateDtPhiBFloatPoint(const int& refLogicLayer,
                                                                  const int& refPhi,
                                                                  const int& refPhiB,
                                                                  unsigned int targetLayer,
                                                                  const int& targetStubPhi,
                                                                  const int& targetStubQuality,
                                                                  const int& targetStubEta,
                                                                  const int& targetStubR,
                                                                  const OMTFConfiguration* omtfConfig) {
  LogTrace("l1tOmtfEventPrint") << "\n"
                                << __FUNCTION__ << ":" << __LINE__ << " refLogicLayer " << refLogicLayer
                                << " targetLayer " << targetLayer << std::endl;
  LogTrace("l1tOmtfEventPrint") << "refPhi " << refPhi << " refPhiB " << refPhiB << " targetStubPhi " << targetStubPhi
                                << " targetStubQuality " << targetStubQuality << std::endl;

  int phiExtr = 0;  //delta phi extrapolated

  float rRefLayer = 431.133;  //[cm], MB1 i.e. refLogicLayer = 0
  if (refLogicLayer == 2)
    rRefLayer = 512.401;  //MB2
  else if (refLogicLayer != 0) {
    return 0;
    //throw cms::Exception("OMTFProcessor<GoldenPatternType>::extrapolateDtPhiB: wrong refStubLogicLayer " + std::to_string(refLogicLayer) );
  }

  int reflLayerIndex = refLogicLayer == 0 ? 0 : 1;

  if (targetLayer == 0 || targetLayer == 2 || targetLayer == 4 || (targetLayer >= 10 && targetLayer <= 14)) {
    //all units are cm. Values from the CMS geometry
    float rTargetLayer = 512.401;  //MB2

    if (targetLayer == 0)
      rTargetLayer = 431.133;  //MB1
    else if (targetLayer == 4)
      rTargetLayer = 617.946;  //MB3

    else if (targetLayer == 10)
      rTargetLayer = 413.675;  //RB1in
    else if (targetLayer == 11)
      rTargetLayer = 448.675;  //RB1out
    else if (targetLayer == 12)
      rTargetLayer = 494.975;  //RB2in
    else if (targetLayer == 13)
      rTargetLayer = 529.975;  //RB2out
    else if (targetLayer == 14)
      rTargetLayer = 602.150;  //RB3

    if (useStubQualInExtr) {
      if (targetLayer == 0 || targetLayer == 2 || targetLayer == 4) {
        if (targetStubQuality == 2 || targetStubQuality == 0)
          rTargetLayer = rTargetLayer - 23.5 / 2;  //inner superlayer
        else if (targetStubQuality == 3 || targetStubQuality == 1)
          rTargetLayer = rTargetLayer + 23.5 / 2;  //outer superlayer
      }
    }

    float d = rTargetLayer - rRefLayer;
    //formula in the form as in the slides explaining the extrapolation algorithm
    //float deltaPhiExtr = d/rTargetLayer * refPhiB / omtfConfig->dtPhiBUnitsRad(); //[rad]
    //phiExtr = round(deltaPhiExtr / omtfConfig->omtfPhiUnit()); //[halfStrip]

    //formula with approximation, used to calculate extrFactor
    float extrFactor = d / rTargetLayer / omtfConfig->dtPhiBUnitsRad() / omtfConfig->omtfPhiUnit();
    phiExtr = extrFactor * (float)refPhiB;  //[halfStrip]

    //formula without approximation
    float deltaPhiExtr = atan(d / rTargetLayer * tan(refPhiB / omtfConfig->dtPhiBUnitsRad()));  //[rad]
    phiExtr = round(deltaPhiExtr / omtfConfig->omtfPhiUnit());                                  //[halfStrip]

    if (useStubQualInExtr & (targetLayer == 0 || targetLayer == 2 || targetLayer == 4)) {
      extrapolFactors[reflLayerIndex][targetLayer][targetStubQuality] = extrFactor;
      extrapolFactorsNorm[reflLayerIndex][targetLayer][targetStubQuality] = 1;
    } else {
      extrapolFactors[reflLayerIndex][targetLayer][0] = extrFactor;
      extrapolFactorsNorm[reflLayerIndex][targetLayer][0] = 1;
    }

    //LogTrace("l1tOmtfEventPrint") <<__FUNCTION__<<":"<<__LINE__<<" deltaPhiExtr "<<deltaPhiExtr<<" phiExtr "<<phiExtr<<std::endl;

    LogTrace("l1tOmtfEventPrint") << "\n"
                                  << __FUNCTION__ << ":" << __LINE__ << " refLogicLayer " << refLogicLayer
                                  << " targetLayer " << std::setw(2) << targetLayer << " targetStubQuality "
                                  << targetStubQuality << " extrFactor " << extrFactor << std::endl;

    LogTrace("l1tOmtfEventPrint") << __FUNCTION__ << ":" << __LINE__ << " refPhiB " << refPhiB << " phiExtr " << phiExtr
                                  << std::endl;

  } else if (targetLayer == 1 || targetLayer == 3 || targetLayer == 5) {
    int deltaPhi = targetStubPhi - refPhi;  //[halfStrip]

    //deltaPhi is here in phi_b hw scale
    deltaPhi = round(deltaPhi * omtfConfig->omtfPhiUnit() * omtfConfig->dtPhiBUnitsRad());

    phiExtr = refPhiB - deltaPhi;  //phiExtr is also in phi_b hw scale
    LogTrace("l1tOmtfEventPrint") << __FUNCTION__ << ":" << __LINE__ << " deltaPhi " << deltaPhi << " phiExtr "
                                  << phiExtr << std::endl;
  } else if ((targetLayer >= 6 && targetLayer <= 9) || (targetLayer >= 15 && targetLayer <= 17)) {
    //if true, for the CSC and endcap RPC the R is taken from the hit coordinates

    float rME = targetStubR;
    if (!useEndcapStubsRInExtr) {
      //all units are cm. This are the average R values for a given chamber (more or less middle of the chamber, but taking into account the OMTF eta range)
      if (targetLayer == 6 || targetLayer == 15)  //ME1/3, RE1/3,
        rME = 600.;
      else if (targetLayer == 7 || targetLayer == 15) {  //ME2/2, RE2/3,
        if (refLogicLayer == 0)
          rME = 600.;
        else
          rME = 640.;
      } else if (targetLayer == 8 || rME == 16) {  //ME3/2, RE3/3,
        if (refLogicLayer == 0)
          rME = 620.;
        else
          rME = 680.;
      } else if (targetLayer == 9) {
        rME = 460.;  //for the refLogicLayer = 1. refLogicLayer = 2 is impossible
      }
    }

    float d = rME - rRefLayer;
    //formula in the form as in the slides explaining the extrapolation algorithm
    //float deltaPhiExtr = d / rME * refPhiB / omtfConfig->dtPhiBUnitsRad();  //[rad]
    //phiExtr = round(deltaPhiExtr / omtfConfig->omtfPhiUnit()); //[halfStrip]

    //formula with approximation, used to calculate extrFactor
    float extrFactor = d / rME / omtfConfig->dtPhiBUnitsRad() / omtfConfig->omtfPhiUnit();
    phiExtr = extrFactor * refPhiB;  //[halfStrip]

    //formula without approximation
    float deltaPhiExtr = atan(d / rME * tan(refPhiB / omtfConfig->dtPhiBUnitsRad()));  //[rad]
    phiExtr = round(deltaPhiExtr / omtfConfig->omtfPhiUnit());                         //[halfStrip]

    if (useEndcapStubsRInExtr) {
      extrapolFactors[reflLayerIndex][targetLayer][abs(targetStubEta)] += extrFactor;
      extrapolFactorsNorm[reflLayerIndex][targetLayer][abs(targetStubEta)]++;
      //extrapolFactors[reflLayerIndex][targetLayer][0] += extrFactor;
      //extrapolFactorsNorm[reflLayerIndex][targetLayer][0]++;
    } else {
      extrapolFactors[reflLayerIndex][targetLayer][0] = extrFactor;
      extrapolFactorsNorm[reflLayerIndex][targetLayer][0] = 1;
    }
    LogTrace("l1tOmtfEventPrint") << "\n"
                                  << __FUNCTION__ << ":" << __LINE__ << " refLogicLayer " << refLogicLayer
                                  << " targetLayer " << std::setw(2) << targetLayer << " targetStubR " << targetStubR
                                  << " targetStubEta " << targetStubEta << " extrFactor "
                                  << " rRefLayer " << rRefLayer << " d " << d << " deltaPhiExtr " << deltaPhiExtr
                                  << " phiExtr " << phiExtr << std::endl;
  }
  //TODO restrict the range of the phiExtr and refPhiB !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  return phiExtr;
}

template <class GoldenPatternType>
int OMTFProcessor<GoldenPatternType>::extrapolateDtPhiBFixedPoint(const int& refLogicLayer,
                                                                  const int& refPhi,
                                                                  const int& refPhiB,
                                                                  unsigned int targetLayer,
                                                                  const int& targetStubPhi,
                                                                  const int& targetStubQuality,
                                                                  const int& targetStubEta,
                                                                  const int& targetStubR,
                                                                  const OMTFConfiguration* omtfConfig) {
  int phiExtr = 0;  //delta phi extrapolated

  int reflLayerIndex = refLogicLayer == 0 ? 0 : 1;
  int extrFactor = 0;

  if (targetLayer == 0 || targetLayer == 2 || targetLayer == 4) {
    if (useStubQualInExtr)
      extrFactor = extrapolFactors[reflLayerIndex][targetLayer][targetStubQuality];
    else
      extrFactor = extrapolFactors[reflLayerIndex][targetLayer][0];
  } else if (targetLayer == 1 || targetLayer == 3 || targetLayer == 5) {
    int deltaPhi = targetStubPhi - refPhi;  //[halfStrip]

    int scaleFactor = this->myOmtfConfig->omtfPhiUnit() * this->myOmtfConfig->dtPhiBUnitsRad() * 512;
    //= 305 for phase-1, 512 is multiplier so that scaleFactor is non-zero integer

    deltaPhi = (deltaPhi * scaleFactor) / 512;  //here deltaPhi is converted to the phi_b hw scale

    phiExtr = refPhiB - deltaPhi;  //phiExtr is also in phi_b hw scale
    //LogTrace("l1tOmtfEventPrint") <<__FUNCTION__<<":"<<__LINE__<<" deltaPhi "<<deltaPhi<<" phiExtr "<<phiExtr<<std::endl;

  } else if (targetLayer >= 10 && targetLayer <= 14) {
    extrFactor = extrapolFactors[reflLayerIndex][targetLayer][0];
  } else if ((targetLayer >= 6 && targetLayer <= 9) || (targetLayer >= 15 && targetLayer <= 17)) {
    if (useEndcapStubsRInExtr) {
      //if given abs(targetStubEta) value is not present in the map, it is added with default value of 0
      //so it should be good. The only problem is that the map can grow...
      extrFactor = extrapolFactors[reflLayerIndex][targetLayer][abs(targetStubEta)];
    } else {
      extrFactor = extrapolFactors[reflLayerIndex][targetLayer][0];
    }
  }

  if (this->myOmtfConfig->isBendingLayer(targetLayer) == false) {
    phiExtr = extrFactor * refPhiB / extrapolMultiplier;
  }

  LogTrace("l1tOmtfEventPrint") << "\n"
                                << __FUNCTION__ << ":" << __LINE__ << " refLogicLayer " << refLogicLayer
                                << " targetLayer " << targetLayer << std::endl;
  LogTrace("l1tOmtfEventPrint") << "refPhi " << refPhi << " refPhiB " << refPhiB << " targetStubPhi " << targetStubPhi
                                << " targetStubQuality " << targetStubQuality << " targetStubEta " << targetStubEta
                                << " extrFactor " << extrFactor << " phiExtr " << phiExtr << std::endl;

  return phiExtr;
}

template <class GoldenPatternType>
int OMTFProcessor<GoldenPatternType>::extrapolateDtPhiB(const MuonStubPtr& refStub,
                                                        const MuonStubPtr& targetStub,
                                                        unsigned int targetLayer,
                                                        const OMTFConfiguration* omtfConfig) {
  if (useFloatingPointExtrapolation)
    return OMTFProcessor<GoldenPatternType>::extrapolateDtPhiBFloatPoint(refStub->logicLayer,
                                                                         refStub->phiHw,
                                                                         refStub->phiBHw,
                                                                         targetLayer,
                                                                         targetStub->phiHw,
                                                                         targetStub->qualityHw,
                                                                         targetStub->etaHw,
                                                                         targetStub->r,
                                                                         omtfConfig);
  return OMTFProcessor<GoldenPatternType>::extrapolateDtPhiBFixedPoint(refStub->logicLayer,
                                                                       refStub->phiHw,
                                                                       refStub->phiBHw,
                                                                       targetLayer,
                                                                       targetStub->phiHw,
                                                                       targetStub->qualityHw,
                                                                       targetStub->etaHw,
                                                                       targetStub->r,
                                                                       omtfConfig);
}
///////////////////////////////////////////////
///////////////////////////////////////////////
//const std::vector<OMTFProcessor::resultsMap> &
template <class GoldenPatternType>
void OMTFProcessor<GoldenPatternType>::processInput(unsigned int iProcessor,
                                                    l1t::tftype mtfType,
                                                    const OMTFinput& aInput,
                                                    std::vector<std::unique_ptr<IOMTFEmulationObserver> >& observers) {
  unsigned int procIndx = this->myOmtfConfig->getProcIndx(iProcessor, mtfType);
  for (auto& itGP : this->theGPs) {
    for (auto& result : itGP->getResults()[procIndx]) {
      result.reset();
    }
  }

  LogTrace("l1tOmtfEventPrint") << __FUNCTION__ << "\n"
                                << __LINE__ << " iProcessor " << iProcessor << " mtfType " << mtfType << " procIndx "
                                << procIndx << " ----------------------" << std::endl;
  //////////////////////////////////////
  //////////////////////////////////////
  std::vector<const RefHitDef*> refHitDefs;

  {
    auto refHitsBits = aInput.getRefHits(iProcessor);
    if (refHitsBits.none())
      return;  // myResults;

    //loop over all possible refHits, e.g. 128
    for (unsigned int iRefHit = 0; iRefHit < this->myOmtfConfig->nRefHits(); ++iRefHit) {
      if (!refHitsBits[iRefHit])
        continue;

      refHitDefs.push_back(&(this->myOmtfConfig->getRefHitsDefs()[iProcessor][iRefHit]));

      if (refHitDefs.size() == this->myOmtfConfig->nTestRefHits())
        break;
    }
  }

  boost::property_tree::ptree procDataTree;
  LogTrace("l1tOmtfEventPrint") << __FUNCTION__ << " " << __LINE__;
  for (unsigned int iLayer = 0; iLayer < this->myOmtfConfig->nLayers(); ++iLayer) {
    //debug
    /*for(auto& h : layerHits) {
      if(h != 5400)
        LogTrace("l1tOmtfEventPrint")<<__FUNCTION__<<" "<<__LINE__<<" iLayer "<<iLayer<<" layerHit "<<h<<std::endl;
    }*/

    for (unsigned int iRefHit = 0; iRefHit < refHitDefs.size(); iRefHit++) {
      const RefHitDef& aRefHitDef = *(refHitDefs[iRefHit]);

      unsigned int refLayerLogicNum = this->myOmtfConfig->getRefToLogicNumber()[aRefHitDef.iRefLayer];
      const MuonStubPtr refStub = aInput.getMuonStub(refLayerLogicNum, aRefHitDef.iInput);
      //int etaRef = refStub->etaHw;

      unsigned int iRegion = aRefHitDef.iRegion;

      MuonStubPtrs1D restrictedLayerStubs = this->restrictInput(iProcessor, iRegion, iLayer, aInput);

      //LogTrace("l1tOmtfEventPrint")<<__FUNCTION__<<" "<<__LINE__<<" iLayer "<<iLayer<<" iRefLayer "<<aRefHitDef.iRefLayer<<std::endl;
      //LogTrace("l1tOmtfEventPrint")<<"iLayer "<<iLayer<<" iRefHit "<<iRefHit;
      //LogTrace("l1tOmtfEventPrint")<<" nTestedRefHits "<<nTestedRefHits<<" aRefHitDef "<<aRefHitDef<<std::endl;

      std::vector<int> extrapolatedPhi(restrictedLayerStubs.size(), 0);

      //TODO make sure the that the iRefLayer numbers used here corresponds to this in the hwToLogicLayer_0x000X.xml
      if ((this->myOmtfConfig->usePhiBExtrapolationMB1() && aRefHitDef.iRefLayer == 0) ||
          (this->myOmtfConfig->usePhiBExtrapolationMB2() && aRefHitDef.iRefLayer == 2)) {
        if ((iLayer != refLayerLogicNum) && (iLayer != refLayerLogicNum + 1)) {
          unsigned int iStub = 0;
          for (auto& targetStub : restrictedLayerStubs) {
            if (targetStub) {
              extrapolatedPhi[iStub] = extrapolateDtPhiB(refStub, targetStub, iLayer, this->myOmtfConfig);

              LogTrace("l1tOmtfEventPrint")
                  << "\n"
                  << __FUNCTION__ << ":" << __LINE__ << " extrapolating from layer " << refLayerLogicNum
                  << " - iRefLayer " << aRefHitDef.iRefLayer << " to layer " << iLayer << " stub " << targetStub
                  << " value " << extrapolatedPhi[iStub] << std::endl;

              if (this->myOmtfConfig->getDumpResultToXML()) {
                auto& extrapolatedPhiTree = procDataTree.add_child("extrapolatedPhi", boost::property_tree::ptree());
                extrapolatedPhiTree.add("<xmlattr>.refLayer", refLayerLogicNum);
                extrapolatedPhiTree.add("<xmlattr>.layer", iLayer);
                extrapolatedPhiTree.add("<xmlattr>.refPhiBHw", refStub->phiBHw);
                extrapolatedPhiTree.add("<xmlattr>.iStub", iStub);
                extrapolatedPhiTree.add("<xmlattr>.qualityHw", targetStub->qualityHw);
                extrapolatedPhiTree.add("<xmlattr>.etaHw", targetStub->etaHw);
                extrapolatedPhiTree.add("<xmlattr>.phiExtr", extrapolatedPhi[iStub]);

                if (this->myOmtfConfig->isBendingLayer(iLayer))
                  extrapolatedPhiTree.add("<xmlattr>.dist_phi", targetStub->phiBHw - extrapolatedPhi[iStub]);
                else
                  extrapolatedPhiTree.add("<xmlattr>.dist_phi", targetStub->phiHw - extrapolatedPhi[iStub]);
              }
            }
            iStub++;
          }
        }
      }

      for (auto& itGP : this->theGPs) {
        if (itGP->key().thePt == 0)  //empty pattern
          continue;

        StubResult stubResult =
            itGP->process1Layer1RefLayer(aRefHitDef.iRefLayer, iLayer, restrictedLayerStubs, extrapolatedPhi, refStub);

        /* LogTrace("l1tOmtfEventPrint")<<__FUNCTION__<<":"<<__LINE__
                                     <<" layerResult: valid"<<stubResult.getValid()
                                     <<" pdfVal "<<stubResult.getPdfVal()
                                     <<std::endl;*/

        itGP->getResults()[procIndx][iRefHit].setStubResult(iLayer, stubResult);
      }
    }
  }

  for (unsigned int iRefHit = 0; iRefHit < refHitDefs.size(); iRefHit++) {
    const RefHitDef& aRefHitDef = *(refHitDefs[iRefHit]);

    unsigned int refLayerLogicNum = this->myOmtfConfig->getRefToLogicNumber()[aRefHitDef.iRefLayer];
    const MuonStubPtr refStub = aInput.getMuonStub(refLayerLogicNum, aRefHitDef.iInput);

    int phiRef = refStub->phiHw;
    int etaRef = refStub->etaHw;

    //calculating the phiExtrp in the case the RefLayer is MB1, to include it in the  candidate phi of candidate
    int phiExtrp = 0;
    if ((this->myOmtfConfig->usePhiBExtrapolationMB1() && aRefHitDef.iRefLayer == 0)) {
      //||(this->myOmtfConfig->getUsePhiBExtrapolationMB2() && aRefHitDef.iRefLayer == 2) ) {  //the extrapolation from the layer 2 to the layer 2 has no sense, so phiExtrp is 0
      LogTrace("l1tOmtfEventPrint") << "\n"
                                    << __FUNCTION__ << ":" << __LINE__
                                    << "extrapolating ref hit to get the phi of the candidate" << std::endl;
      if (useFloatingPointExtrapolation)
        phiExtrp = extrapolateDtPhiBFloatPoint(
            aRefHitDef.iRefLayer, phiRef, refStub->phiBHw, 2, 0, 6, 0, 0, this->myOmtfConfig);
      else
        phiExtrp = extrapolateDtPhiBFixedPoint(
            aRefHitDef.iRefLayer, phiRef, refStub->phiBHw, 2, 0, 6, 0, 0, this->myOmtfConfig);
    }

    for (auto& itGP : this->theGPs) {
      if (itGP->key().thePt == 0)  //empty pattern
        continue;

      int phiRefSt2 = itGP->propagateRefPhi(phiRef + phiExtrp, etaRef, aRefHitDef.iRefLayer);
      itGP->getResults()[procIndx][iRefHit].set(aRefHitDef.iRefLayer, phiRefSt2, etaRef, phiRef);
    }
  }

  //////////////////////////////////////
  //////////////////////////////////////
  {
    for (auto& itGP : this->theGPs) {
      itGP->finalise(procIndx);
      //debug
      /*for(unsigned int iRefHit = 0; iRefHit < itGP->getResults()[procIndx].size(); ++iRefHit) {
        if(itGP->getResults()[procIndx][iRefHit].isValid()) {
          LogTrace("l1tOmtfEventPrint")<<__FUNCTION__<<":"<<"__LINE__"<<itGP->getResults()[procIndx][iRefHit]<<std::endl;
        }
      }*/
    }
  }

  for (auto& obs : observers)
    obs->addProcesorData("extrapolation", procDataTree);

  return;
}
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////

template <class GoldenPatternType>
std::vector<l1t::RegionalMuonCand> OMTFProcessor<GoldenPatternType>::run(
    unsigned int iProcessor,
    l1t::tftype mtfType,
    int bx,
    OMTFinputMaker* inputMaker,
    std::vector<std::unique_ptr<IOMTFEmulationObserver> >& observers) {
  //uncomment if you want to check execution time of each method
  //boost::timer::auto_cpu_timer t("%ws wall, %us user in getProcessorCandidates\n");

  for (auto& obs : observers)
    obs->observeProcesorBegin(iProcessor, mtfType);

  //input is shared_ptr because the observers may need them after the run() method execution is finished
  std::shared_ptr<OMTFinput> input = std::make_shared<OMTFinput>(this->myOmtfConfig);
  inputMaker->buildInputForProcessor(input->getMuonStubs(), iProcessor, mtfType, bx, bx, observers);

  if (this->myOmtfConfig->cleanStubs()) {
    //this has sense for the pattern generation from the tracks with the secondaries
    //if more than one stub is in a given layer, all stubs are removed from this layer
    for (unsigned int iLayer = 0; iLayer < input->getMuonStubs().size(); ++iLayer) {
      auto& layerStubs = input->getMuonStubs()[iLayer];
      int count = std::count_if(layerStubs.begin(), layerStubs.end(), [](auto& ptr) { return ptr != nullptr; });
      if (count > 1) {
        for (auto& ptr : layerStubs)
          ptr.reset();

        LogTrace("OMTFReconstruction") << __FUNCTION__ << ":" << __LINE__ << "cleaning stubs in the layer " << iLayer
                                       << " stubs count :" << count << std::endl;
      }
    }
  }

  //LogTrace("l1tOmtfEventPrint")<<"buildInputForProce "; t.report();
  processInput(iProcessor, mtfType, *(input.get()), observers);

  //LogTrace("l1tOmtfEventPrint")<<"processInput       "; t.report();
  AlgoMuons algoCandidates = sortResults(iProcessor, mtfType);

  if (ptAssignment) {
    for (auto& myCand : algoCandidates) {
      if (myCand->isValid()) {
        auto pts = ptAssignment->getPts(myCand, observers);
        /*for (unsigned int i = 0; i < pts.size(); i++) {
        trackAddr[10 + i] = this->myOmtfConfig->ptGevToHw(pts[i]);
      }*/
      }
    }
  }

  //LogTrace("l1tOmtfEventPrint")<<"sortResults        "; t.report();
  // perform GB
  //watch out: etaBits2HwEta is used in the ghostBust to convert the AlgoMuons eta, it affect algoCandidates as they are pointers
  AlgoMuons gbCandidates = ghostBust(algoCandidates);

  //LogTrace("l1tOmtfEventPrint")<<"ghostBust"; t.report();
  // fill RegionalMuonCand colleciton
  std::vector<l1t::RegionalMuonCand> candMuons = getFinalcandidates(iProcessor, mtfType, gbCandidates);

  //LogTrace("l1tOmtfEventPrint")<<"getFinalcandidates "; t.report();
  //fill outgoing collection
  for (auto& candMuon : candMuons) {
    candMuon.setHwQual(candMuon.hwQual());
  }

  for (auto& obs : observers) {
    obs->observeProcesorEmulation(iProcessor, mtfType, input, algoCandidates, gbCandidates, candMuons);
  }

  return candMuons;
}

template <class GoldenPatternType>
void OMTFProcessor<GoldenPatternType>::printInfo() const {
  edm::LogVerbatim("OMTFReconstruction") << __PRETTY_FUNCTION__ << std::endl;

  ProcessorBase<GoldenPatternType>::printInfo();
}

template <class GoldenPatternType>
void OMTFProcessor<GoldenPatternType>::saveExtrapolFactors() {
  //if(this->myOmtfConfig->nProcessors() == 3) //phase2
  extrapolMultiplier = 512;

  boost::property_tree::ptree tree;
  auto& extrFactorsTree = tree.add("ExtrapolationFactors", "");
  extrFactorsTree.add("<xmlattr>.multiplier", extrapolMultiplier);

  edm::LogVerbatim("OMTFReconstruction") << "saving extrapolFactors to ExtrapolationFactors.xml" << std::endl;
  for (unsigned int iRefLayer = 0; iRefLayer < extrapolFactors.size(); iRefLayer++) {
    for (unsigned int iLayer = 0; iLayer < extrapolFactors[iRefLayer].size(); iLayer++) {
      edm::LogVerbatim("OMTFReconstruction") << " iRefLayer " << iRefLayer << " iLayer " << iLayer << std::endl;

      auto& layerTree = extrFactorsTree.add_child("Lut", boost::property_tree::ptree());
      layerTree.add("<xmlattr>.RefLayer", std::to_string(iRefLayer));
      layerTree.add("<xmlattr>.Layer", iLayer);

      if (useStubQualInExtr && (iLayer == 0 || iLayer == 2 || iLayer == 4))
        layerTree.add("<xmlattr>.KeyType", "quality");
      else if (useEndcapStubsRInExtr && ((iLayer >= 6 && iLayer <= 9) || (iLayer >= 15 && iLayer <= 17)))
        layerTree.add("<xmlattr>.KeyType", "eta");
      else
        layerTree.add("<xmlattr>.KeyType", "none");

      for (auto& extrFactors : extrapolFactors[iRefLayer][iLayer]) {
        int norm = 1;
        if (!extrapolFactorsNorm[iRefLayer][iLayer].empty())
          norm = extrapolFactorsNorm[iRefLayer][iLayer][extrFactors.first];
        auto& lutVal = layerTree.add_child("LutVal", boost::property_tree::ptree());
        if (useEndcapStubsRInExtr && ((iLayer >= 6 && iLayer <= 9) || (iLayer >= 15 && iLayer <= 17)))
          lutVal.add("<xmlattr>.key", extrFactors.first);
        else
          lutVal.add("<xmlattr>.key", extrFactors.first);

        double value = round(extrapolMultiplier * extrFactors.second / norm);
        lutVal.add("<xmlattr>.value", value);

        edm::LogVerbatim("OMTFReconstruction")
            << std::setw(4) << " key = " << extrFactors.first << " extrFactors.second " << std::setw(10)
            << extrFactors.second << " norm " << std::setw(6) << norm << " value/norm " << std::setw(10)
            << extrFactors.second / norm << " value " << value << std::endl;
      }
    }
  }

  boost::property_tree::write_xml("ExtrapolationFactors.xml",
                                  tree,
                                  std::locale(),
                                  boost::property_tree::xml_parser::xml_writer_make_settings<std::string>(' ', 2));
}

template <class GoldenPatternType>
void OMTFProcessor<GoldenPatternType>::loadExtrapolFactors(const std::string& filename) {
  boost::property_tree::ptree tree;

  boost::property_tree::read_xml(filename, tree);

  edm::LogVerbatim("OMTFReconstruction") << "loadExtrapolFactors from file " << filename << std::endl;

  extrapolMultiplier = tree.get<int>("ExtrapolationFactors.<xmlattr>.multiplier");
  edm::LogVerbatim("OMTFReconstruction") << "extrapolMultiplier " << extrapolMultiplier << std::endl;

  auto& lutNodes = tree.get_child("ExtrapolationFactors");
  for (boost::property_tree::ptree::value_type& lutNode : lutNodes) {
    if (lutNode.first == "Lut") {
      int iRefLayer = lutNode.second.get<int>("<xmlattr>.RefLayer");
      int iLayer = lutNode.second.get<int>("<xmlattr>.Layer");
      std::string keyType = lutNode.second.get<std::string>("<xmlattr>.KeyType");

      edm::LogVerbatim("OMTFReconstruction")
          << "iRefLayer " << iRefLayer << " iLayer " << iLayer << " keyType " << keyType << std::endl;

      auto& valueNodes = lutNode.second;
      for (boost::property_tree::ptree::value_type& valueNode : valueNodes) {
        if (valueNode.first == "LutVal") {
          int key = valueNode.second.get<int>("<xmlattr>.key");
          float value = valueNode.second.get<float>("<xmlattr>.value");
          extrapolFactors.at(iRefLayer).at(iLayer)[key] = value;
          edm::LogVerbatim("OMTFReconstruction") << "key " << key << " value " << value << std::endl;
        }
      }
    }
  }
}

/////////////////////////////////////////////////////////

template class OMTFProcessor<GoldenPattern>;
template class OMTFProcessor<GoldenPatternWithStat>;
template class OMTFProcessor<GoldenPatternWithThresh>;
