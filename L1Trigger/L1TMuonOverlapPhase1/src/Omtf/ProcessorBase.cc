/*
 * ProcessorBase.cpp
 *
 *  Created on: Jul 28, 2017
 *      Author: kbunkow
 */

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/ProcessorBase.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPattern.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPatternWithStat.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/XMLConfigWriter.h"

#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"
#include "SimDataFormats/Track/interface/SimTrack.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

template <class GoldenPatternType>
ProcessorBase<GoldenPatternType>::ProcessorBase(OMTFConfiguration* omtfConfig,
                                                GoldenPatternVec<GoldenPatternType>&& gps)
    : myOmtfConfig(omtfConfig), theGPs(std::move(gps)) {
  for (auto& gp : theGPs) {
    gp->setConfig(myOmtfConfig);
  }

  initPatternPtRange(true);

  //initPatternPtRange(true); is called in the setGPs
  omtfConfig->setPatternPtRange(getPatternPtRange());
};

template <class GoldenPatternType>
void ProcessorBase<GoldenPatternType>::resetConfiguration() {
  theGPs.clear();
}

///////////////////////////////////////////////
///////////////////////////////////////////////
template <class GoldenPatternType>
bool ProcessorBase<GoldenPatternType>::configure(OMTFConfiguration* omtfConfig,
                                                 const L1TMuonOverlapParams* omtfPatterns) {
  resetConfiguration();

  myOmtfConfig = omtfConfig;

  const l1t::LUT* chargeLUT = omtfPatterns->chargeLUT();
  const l1t::LUT* etaLUT = omtfPatterns->etaLUT();
  const l1t::LUT* ptLUT = omtfPatterns->ptLUT();
  const l1t::LUT* pdfLUT = omtfPatterns->pdfLUT();
  const l1t::LUT* meanDistPhiLUT = omtfPatterns->meanDistPhiLUT();
  const l1t::LUT* distPhiShiftLUT = omtfPatterns->distPhiShiftLUT();

  unsigned int nGPs = myOmtfConfig->nGoldenPatterns();
  edm::LogVerbatim("OMTFReconstruction")
      << "ProcessorBase<>::configure. Building patterns from L1TMuonOverlapParams (LUTs). nGoldenPatterns() " << nGPs
      << std::endl;

  unsigned int address = 0;
  unsigned int iEta, iPt;
  int iCharge;
  //int meanDistPhiSize = myOmtfConfig->nLayers() * myOmtfConfig->nRefLayers() * myOmtfConfig->nGoldenPatterns();

  unsigned int group = 0;
  unsigned int indexInGroup = 0;
  for (unsigned int iGP = 0; iGP < nGPs; ++iGP) {
    address = iGP;
    iEta = etaLUT->data(address);
    iCharge = chargeLUT->data(address) == 0 ? -1 : 1;
    iPt = ptLUT->data(address);

    //the patterns in the LUTs should contain the empty patterns, only then the group and indexInGroup calculation works
    group = iGP / myOmtfConfig->patternsInGroup;
    indexInGroup = iGP % myOmtfConfig->patternsInGroup + 1;
    Key aKey(iEta, iPt, iCharge, theGPs.size(), group, indexInGroup);
    if (iPt == 0) {
      LogTrace("OMTFReconstruction")
          << "skipping empty pattern " << aKey << " "
          << std::endl;  //<<myOmtfConfig->getPatternPtRange(iGP).ptFrom<<" - "<<myOmtfConfig->getPatternPtRange(iGP).ptTo<<" GeV"<<std::endl; PatternPtRange is not initialized here yet!!!!
      continue;
    }

    LogTrace("OMTFReconstruction")
        << "adding pattern " << aKey << " "
        << std::endl;  //<<myOmtfConfig->getPatternPtRange(iGP).ptFrom<<" - "<<myOmtfConfig->getPatternPtRange(iGP).ptTo<<" GeV"<<std::endl; PatternPtRange is not initialized here yet!!!!

    GoldenPatternType* aGP = new GoldenPatternType(aKey, myOmtfConfig);

    for (unsigned int iLayer = 0; iLayer < myOmtfConfig->nLayers(); ++iLayer) {
      for (unsigned int iRefLayer = 0; iRefLayer < myOmtfConfig->nRefLayers(); ++iRefLayer) {
        address = iRefLayer + iLayer * myOmtfConfig->nRefLayers() +
                  iGP * (myOmtfConfig->nRefLayers() * myOmtfConfig->nLayers());

        ///Mean dist phi data
        //LUT values are only positive, therefore to have negative  meanDistPh half of the max LUT value is subtracted
        int value = meanDistPhiLUT->data(address) - (1 << (meanDistPhiLUT->nrBitsData() - 1));
        aGP->setMeanDistPhiValue(value, iLayer, iRefLayer, 0);

        /* uncomment this if you need  meanDistPhi1 in the LUTs (and set useMeanDistPhi1 in XMLConfigReader::readLUTs)
        if ( (1 << meanDistPhiLUT->nrBitsAddress()) > 2 * meanDistPhiSize ) {
          //for the  version of the meanDistPhi which have two values for each gp,iLayer,iRefLayer, FIXME: do it a better way
          value = meanDistPhiLUT->data(address + meanDistPhiSize) - (1 << (meanDistPhiLUT->nrBitsData() - 1));
          //the second meanDistPhi is in the LUT at the position (address+meanDistPhiSize)
          aGP->setMeanDistPhiValue(value, iLayer, iRefLayer, 1);
        }*/

        //selDistPhiShift, if the distPhiShiftLUT is nullptr, it means it was not present in the L1TMuonOverlapParamsRcd
        if (distPhiShiftLUT) {
          value = distPhiShiftLUT->data(address);  //distPhiShiftLUT values are only positive
          aGP->setDistPhiBitShift(value, iLayer, iRefLayer);
        }
      }
      ///Pdf data
      for (unsigned int iRefLayer = 0; iRefLayer < myOmtfConfig->nRefLayers(); ++iRefLayer) {
        for (unsigned int iPdf = 0; iPdf < (unsigned int)(1 << myOmtfConfig->nPdfAddrBits()); ++iPdf) {
          address = iPdf + iRefLayer * (1 << myOmtfConfig->nPdfAddrBits()) +
                    iLayer * myOmtfConfig->nRefLayers() * (1 << myOmtfConfig->nPdfAddrBits()) +
                    iGP * myOmtfConfig->nLayers() * myOmtfConfig->nRefLayers() * (1 << myOmtfConfig->nPdfAddrBits());
          int value = pdfLUT->data(address);  //here only int is possible
          aGP->setPdfValue(value, iLayer, iRefLayer, iPdf);

          //edm::LogVerbatim("OMTFReconstruction")<<" iLayer "<<iLayer<<" iRefLayer "<<iRefLayer<<" iPdf "<<iPdf << " address "<<address<<" value "<<value<<std::endl;
        }
      }
    }

    addGP(aGP);
  }

  initPatternPtRange(true);

  omtfConfig->setPatternPtRange(getPatternPtRange());

  return true;
}

///////////////////////////////////////////////
///////////////////////////////////////////////
template <class GoldenPatternType>
void ProcessorBase<GoldenPatternType>::addGP(GoldenPatternType* aGP) {
  theGPs.emplace_back(std::unique_ptr<GoldenPatternType>(aGP));
}

////////////////////////////////////////////
////////////////////////////////////////////
template <class GoldenPatternType>
MuonStubPtrs1D ProcessorBase<GoldenPatternType>::restrictInput(unsigned int iProcessor,
                                                               unsigned int iRegion,
                                                               unsigned int iLayer,
                                                               const OMTFinput& input) {
  MuonStubPtrs1D layerStubs;

  unsigned int iStart = myOmtfConfig->getConnections()[iProcessor][iRegion][iLayer].first;
  unsigned int iEnd = iStart + myOmtfConfig->getConnections()[iProcessor][iRegion][iLayer].second - 1;

  for (unsigned int iInput = 0; iInput < input.getMuonStubs()[iLayer].size(); ++iInput) {
    if (iInput >= iStart && iInput <= iEnd) {
      if (this->myOmtfConfig->isBendingLayer(iLayer)) {
        layerStubs.push_back(input.getMuonStub(iLayer - 1, iInput));
      } else
        layerStubs.push_back(input.getMuonStub(iLayer, iInput));  //input.getHitPhi(iLayer, iInput)
    }
  }
  //std::cout<<__FUNCTION__<<":"<<__LINE__<<" layerHits.size() "<<layerHits.size()<<std::endl;
  return layerStubs;
}

////////////////////////////////////////////
////////////////////////////////////////////
template <class GoldenPatternType>
void ProcessorBase<GoldenPatternType>::initPatternPtRange(bool firstPatFrom0) {
  patternPts.clear();

  bool firstPos = firstPatFrom0;
  bool firstNeg = firstPatFrom0;
  for (unsigned int iPat = 0; iPat < theGPs.size(); iPat++) {
    OMTFConfiguration::PatternPt patternPt;
    int charge = theGPs[iPat]->key().theCharge;
    if (theGPs[iPat] == nullptr || theGPs[iPat]->key().thePt == 0) {
      patternPts.push_back(patternPt);
      continue;
    }

    patternPt.ptFrom = myOmtfConfig->hwPtToGev(theGPs[iPat]->key().thePt);
    if (firstPos && theGPs[iPat]->key().theCharge == 1) {
      patternPt.ptFrom = 0;
      firstPos = false;
    }
    if (firstNeg && theGPs[iPat]->key().theCharge == -1) {
      patternPt.ptFrom = 0;
      firstNeg = false;
    }

    unsigned int iPat1 = iPat;
    while (true) {  //to skip the empty patterns with pt=0 and patterns with opposite charge
      iPat1++;
      if (iPat1 == theGPs.size())
        break;
      if (theGPs[iPat1]->key().thePt != 0 && theGPs[iPat1]->key().theCharge == charge)
        break;
    }

    if (iPat1 == theGPs.size())
      patternPt.ptTo = 10000;  //inf
    else
      patternPt.ptTo = myOmtfConfig->hwPtToGev(theGPs[iPat1]->key().thePt);

    patternPt.charge = charge;
    patternPts.push_back(patternPt);
  }

  /*  for(unsigned int iPat = 0; iPat < theGPs.size(); iPat++) {
    std::cout<<theGPs[iPat]->key()<<" ptFrom "<<patternPts[iPat].ptFrom<<" ptFrom "<<patternPts[iPat].ptTo<<std::endl;
  }*/

  edm::LogTrace_("OMTFReconstruction") << __FUNCTION__ << ":" << __LINE__ << " patternPts.size() " << patternPts.size()
                                       << endl;
}

template <class GoldenPatternType>
void ProcessorBase<GoldenPatternType>::printInfo() const {
  myOmtfConfig->printConfig();

  edm::LogVerbatim("OMTFReconstruction") << "\npatterns:" << std::endl;
  unsigned int patNum = 0;
  for (auto& gp : theGPs) {
    edm::LogVerbatim("OMTFReconstruction")
        << std::setw(2) << patNum << " " << gp->key() << " " << myOmtfConfig->getPatternPtRange(patNum).ptFrom << " - "
        << myOmtfConfig->getPatternPtRange(patNum).ptTo << " GeV" << std::endl;
    patNum++;
  }

  /* //can be useful for debug, uncomment if needed
  XMLConfigWriter xmlWriter(this->myOmtfConfig, false, false);
  xmlWriter.writeGPs(this->theGPs, "patternsAsInTheProcessor.xml");*/
}

//to force compiler to compile the above methods with needed GoldenPatterns types
template class ProcessorBase<GoldenPattern>;
template class ProcessorBase<GoldenPatternWithThresh>;
template class ProcessorBase<GoldenPatternWithStat>;
