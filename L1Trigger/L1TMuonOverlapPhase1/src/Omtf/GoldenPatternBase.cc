/*
 * GoldenPatternBase.cpp
 *
 *  Created on: Oct 3, 2017
 *      Author: kbunkow
 */

#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPatternBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iomanip>

std::ostream& operator<<(std::ostream& out, const Key& o) {
  out << "Key_" << std::setw(2) << o.theNumber << " hwNum " << std::setw(2) << o.getHwPatternNumber() << " group "
      << std::setw(2) << o.theGroup << ":" << o.theIndexInGroup << " : (eta=" << o.theEtaCode << ", pt=" << std::setw(3)
      << o.thePt << ", charge=" << setw(2) << o.theCharge << ")";
  return out;
}

GoldenPatternBase::GoldenPatternBase(const Key& aKey) : theKey(aKey), myOmtfConfig(nullptr) {
  //std::cout<<__FUNCTION__<<":"<<__LINE__<<std::endl;
}

GoldenPatternBase::GoldenPatternBase(const Key& aKey, const OMTFConfiguration* omtfConfig)
    : theKey(aKey),
      myOmtfConfig(omtfConfig),
      results(boost::extents[myOmtfConfig->processorCnt()][myOmtfConfig->nTestRefHits()]) {
  //std::cout<<__FUNCTION__<<":"<<__LINE__<<std::endl;
  for (unsigned int iProc = 0; iProc < results.size(); iProc++) {
    for (unsigned int iTestRefHit = 0; iTestRefHit < results[iProc].size(); iTestRefHit++) {
      results[iProc][iTestRefHit].init(omtfConfig);
    }
  }
}

void GoldenPatternBase::setConfig(const OMTFConfiguration* omtfConfig) {
  myOmtfConfig = omtfConfig;
  results.resize(boost::extents[myOmtfConfig->processorCnt()][myOmtfConfig->nTestRefHits()]);
  for (unsigned int iProc = 0; iProc < results.size(); iProc++) {
    for (unsigned int iTestRefHit = 0; iTestRefHit < results[iProc].size(); iTestRefHit++) {
      results[iProc][iTestRefHit].init(omtfConfig);
    }
  }
}

////////////////////////////////////////////////////
////////////////////////////////////////////////////
StubResult GoldenPatternBase::process1Layer1RefLayer(unsigned int iRefLayer,
                                                     unsigned int iLayer,
                                                     MuonStubPtrs1D layerStubs,
                                                     const MuonStubPtr refStub) {
  //if (this->getDistPhiBitShift(iLayer, iRefLayer) != 0) std::cout<<__FUNCTION__<<":"<<__LINE__<<key()<<this->getDistPhiBitShift(iLayer, iRefLayer)<<std::endl;
  //GoldenPatternResult::LayerResult aResult(0, 0, 0, 0); //0, 0

  int phiMean = this->meanDistPhiValue(iLayer, iRefLayer, refStub->phiBHw);
  int phiDistMin = myOmtfConfig->nPhiBins();  //1<<(myOmtfConfig->nPdfAddrBits()); //"infinite" value for the beginning

  ///Select hit closest to the mean of probability
  ///distribution in given layer
  MuonStubPtr selectedStub;

  int phiRefHit = 0;
  if (refStub)
    phiRefHit = refStub->phiHw;

  if (this->myOmtfConfig->isBendingLayer(iLayer)) {
    phiRefHit = 0;  //phi ref hit for the banding layer set to 0, since it should not be included in the phiDist
  }

  for (auto& stub : layerStubs) {
    if (!stub)  //empty pointer
      continue;

    int hitPhi = stub->phiHw;
    if (this->myOmtfConfig->isBendingLayer(iLayer)) {
      //if (stub->qualityHw < this->myOmtfConfig->getMinDtPhiBQuality()) //moved to OMTFInputMaker
      //  continue;  //rejecting phiB of the low quality DT stubs

      hitPhi = stub->phiBHw;
    }

    if (hitPhi >= (int)myOmtfConfig->nPhiBins())  //TODO is this needed now? the empty hit will be empty stub
      continue;  //empty itHits are marked with nPhiBins() in OMTFProcessor::restrictInput

    int phiDist = this->myOmtfConfig->foldPhi(hitPhi - phiMean - phiRefHit);
    //for standard omtf foldPhi is not needed, but if one processor works for full phi then it is
    //if (this->getDistPhiBitShift(iLayer, iRefLayer) != 0)
    /*edm::LogVerbatim("l1tOmtfEventPrint") <<"\n"<<__FUNCTION__<<":"<<__LINE__<<" "<<theKey<<std::endl;
    edm::LogVerbatim("l1tOmtfEventPrint") <<__FUNCTION__<<":"<<__LINE__
    		<<"  iRefLayer "<<iRefLayer<<" iLayer "<<iLayer
    		<<" hitPhi "<<hitPhi<<" phiMean "<<phiMean<<" phiRefHit "<<phiRefHit<<" phiDist "<<phiDist<<std::endl;*/

    //firmware works on the sign-value, shift must be done on abs(phiDist)
    int sign = phiDist < 0 ? -1 : 1;
    phiDist = abs(phiDist) >> this->getDistPhiBitShift(iLayer, iRefLayer);
    phiDist *= sign;
    //if the shift is done here, it means that the phiMean in the xml should be the same as without shift
    //if (this->getDistPhiBitShift(iLayer, iRefLayer) != 0) std::cout<<__FUNCTION__<<":"<<__LINE__<<" phiDist "<<phiDist<<std::endl;
    if (abs(phiDist) < abs(phiDistMin)) {
      phiDistMin = phiDist;
      selectedStub = stub;
    }
  }

  if (!selectedStub) {
    if (this->myOmtfConfig->isNoHitValueInPdf()) {
      PdfValueType pdfVal = this->pdfValue(iLayer, iRefLayer, 0);
      return StubResult(pdfVal, false, myOmtfConfig->nPhiBins(), iLayer, selectedStub);
    } else {
      return StubResult(0, false, myOmtfConfig->nPhiBins(), iLayer, selectedStub);  //2018 version
    }
  }

  int pdfMiddle = 1 << (myOmtfConfig->nPdfAddrBits() - 1);

  /*  debug
  if(phiDistMin != 128 && iRefLayer == 0 && iLayer == 1)*/
  /*edm::LogVerbatim("l1tOmtfEventPrint") <<__FUNCTION__<<":"<<__LINE__<<" iRefLayer "<<iRefLayer<<" iLayer "<<iLayer<<" selectedStub "<<*selectedStub
		  <<" phiDistMin "<<phiDistMin<<" phiMean "<<phiMean<<" shift "<<this->getDistPhiBitShift(iLayer, iRefLayer)<<std::endl;*/

  ///Check if phiDistMin is within pdf range -63 +63
  ///in firmware here the arithmetic "value and sign" is used, therefore the range is -63 +63, and not -64 +63
  if (abs(phiDistMin) > ((1 << (myOmtfConfig->nPdfAddrBits() - 1)) - 1)) {
    return StubResult(0, false, phiDistMin + pdfMiddle, iLayer, selectedStub);

    //return GoldenPatternResult::LayerResult(this->pdfValue(iLayer, iRefLayer, 0), false, phiDistMin + pdfMiddle, selHit);
    //in some algorithms versions with thresholds we use the bin 0 to store the pdf value returned when there was no hit.
    //in the version without thresholds, the value in the bin 0 should be 0
  }

  ///Shift phidist, so 0 is at the middle of the range
  phiDistMin += pdfMiddle;
  //if (this->getDistPhiBitShift(iLayer, iRefLayer) != 0) std::cout<<__FUNCTION__<<":"<<__LINE__<<" phiDistMin "<<phiDistMin<<std::endl;
  PdfValueType pdfVal = this->pdfValue(iLayer, iRefLayer, phiDistMin);
  if (pdfVal <= 0) {
    return StubResult(0, false, phiDistMin, iLayer, selectedStub);
    //return GoldenPatternResult::LayerResult(this->pdfValue(iLayer, iRefLayer, 0), false, phiDistMin, selHit); //the pdf[0] needed in some versions of algorithm with threshold
  }
  return StubResult(pdfVal, true, phiDistMin, iLayer, selectedStub);
}

////////////////////////////////////////////////////
////////////////////////////////////////////////////
void GoldenPatternBase::finalise(unsigned int procIndx) {
  for (auto& result : getResults()[procIndx]) {
    result.finalise();
  }
}
