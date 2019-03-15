/*
 * GoldenPatternBase.cpp
 *
 *  Created on: Oct 3, 2017
 *      Author: kbunkow
 */


#include <L1Trigger/L1TMuonBayes/interface/Omtf/GoldenPatternBase.h>

GoldenPatternBase::GoldenPatternBase(const Key & aKey) : theKey(aKey), myOmtfConfig(0) {
  //std::cout<<__FUNCTION__<<":"<<__LINE__<<std::endl;
}

GoldenPatternBase::GoldenPatternBase(const Key& aKey, const OMTFConfiguration * omtfConfig) : theKey(aKey), myOmtfConfig(omtfConfig),
results(boost::extents[myOmtfConfig->processorCnt()][myOmtfConfig->nTestRefHits()]) {
  //std::cout<<__FUNCTION__<<":"<<__LINE__<<std::endl;
  for(unsigned int iProc = 0; iProc < results.size(); iProc++) {
    for(unsigned int iTestRefHit = 0; iTestRefHit < results[iProc].size(); iTestRefHit++) {
      results[iProc][iTestRefHit].init(omtfConfig);
    }
  }
}

void GoldenPatternBase::setConfig(const OMTFConfiguration * omtfConfig) {
  myOmtfConfig = omtfConfig;
  results.resize(boost::extents[myOmtfConfig->processorCnt()][myOmtfConfig->nTestRefHits()]);
  for(unsigned int iProc = 0; iProc < results.size(); iProc++) {
    for(unsigned int iTestRefHit = 0; iTestRefHit < results[iProc].size(); iTestRefHit++) {
      results[iProc][iTestRefHit].init(omtfConfig);
    }
  }
}

////////////////////////////////////////////////////
////////////////////////////////////////////////////
StubResult GoldenPatternBase::process1Layer1RefLayer(unsigned int iRefLayer,
    unsigned int iLayer,
    MuonStubPtrs1D layerStubs,
    const MuonStubPtr refStub)
{
  //if (this->getDistPhiBitShift(iLayer, iRefLayer) != 0) std::cout<<__FUNCTION__<<":"<<__LINE__<<key()<<this->getDistPhiBitShift(iLayer, iRefLayer)<<std::endl;
  //GoldenPatternResult::LayerResult aResult(0, 0, 0, 0); //0, 0

  int phiMean = this->meanDistPhiValue(iLayer, iRefLayer, refStub->phiBHw);
  int phiDistMin = myOmtfConfig->nPhiBins(); //1<<(myOmtfConfig->nPdfAddrBits()); //"infinite" value for the beginning

  ///Select hit closest to the mean of probability
  ///distribution in given layer
  MuonStubPtr selectedStub;
  for(auto& stub: layerStubs){
    if(!stub) //empty pointer
      continue;

    int hitPhi = stub->phiHw;
    int phiRefHit = 0;
    if(refStub)
      phiRefHit = refStub->phiHw;

    if(this->myOmtfConfig->isBendingLayer(iLayer) ) {
      hitPhi = stub->phiBHw;
      phiRefHit = 0; //phi ref hit for the banding layer set to 0, since it should not be included in the phiDist
    }
    if(hitPhi >= (int)myOmtfConfig->nPhiBins()) //TODO is this needed now? the empty hit will be empty stub
      continue;  //empty itHits are marked with nPhiBins() in OMTFProcessor::restrictInput

    int phiDist = this->myOmtfConfig->foldPhi(hitPhi - phiMean - phiRefHit); //for standard omtf foldPhi is not needeed, but if one processor works for full phi then it is
    //if (this->getDistPhiBitShift(iLayer, iRefLayer) != 0)
    //std::cout<<__FUNCTION__<<":"<<__LINE__<<" itHit "<<itHit<<" phiMean "<<phiMean<<" phiRefHit "<<phiRefHit<<" phiDist "<<phiDist<<std::endl;
    phiDist = phiDist >> this->getDistPhiBitShift(iLayer, iRefLayer); //N.B. >> works well also for negative nnumbers. NB2. if the shift is done here, it means that the phiMean in the xml should be the same as without shift
    //if (this->getDistPhiBitShift(iLayer, iRefLayer) != 0) std::cout<<__FUNCTION__<<":"<<__LINE__<<" phiDist "<<phiDist<<std::endl;
    if(abs(phiDist) < abs(phiDistMin)) {
      phiDistMin = phiDist;
      selectedStub = stub;
    }
  }

  if(!selectedStub) {
    return StubResult(0, false, myOmtfConfig->nPhiBins(), iLayer, selectedStub);
  }

  int pdfMiddle = 1<<(myOmtfConfig->nPdfAddrBits()-1);

/*  debug
  if(phiDistMin != 128 && iRefLayer == 0 && iLayer == 1)
    std::cout<<__FUNCTION__<<":"<<__LINE__<<" iRefLayer "<<iRefLayer<<" iLayer "<<iLayer<<" selHit "<<selHit<<" phiDistMin "
      <<phiDistMin<<" phiMean "<<phiMean<<" shift "<<this->getDistPhiBitShift(iLayer, iRefLayer)<<std::endl;*/

  ///Check if phiDistMin is within pdf range -63 +63
  ///in firmware here the arithmetic "value and sign" is used, therefore the range is -63 +63, and not -64 +63
  if(abs(phiDistMin) > ( (1<<(myOmtfConfig->nPdfAddrBits()-1)) -1) ) {
    return StubResult(0, false, phiDistMin + pdfMiddle, iLayer, selectedStub);

    //return GoldenPatternResult::LayerResult(this->pdfValue(iLayer, iRefLayer, 0), false, phiDistMin + pdfMiddle, selHit);
    //in some algorithms versions with thresholds we use the bin 0 to store the pdf value returned when there was no hit.
    //in the version without thresholds, the value in the bin 0 should be 0
  }

  ///Shift phidist, so 0 is at the middle of the range
  phiDistMin += pdfMiddle;
  //if (this->getDistPhiBitShift(iLayer, iRefLayer) != 0) std::cout<<__FUNCTION__<<":"<<__LINE__<<" phiDistMin "<<phiDistMin<<std::endl;
  PdfValueType pdfVal = this->pdfValue(iLayer, iRefLayer, phiDistMin);
  if(pdfVal <= 0) {
    return StubResult(0, false, phiDistMin, iLayer, selectedStub);
    //return GoldenPatternResult::LayerResult(this->pdfValue(iLayer, iRefLayer, 0), false, phiDistMin, selHit); //the pdf[0] needed in some versions of algorithm with threshold
  }
  return StubResult(pdfVal, true, phiDistMin, iLayer, selectedStub);
}

////////////////////////////////////////////////////
////////////////////////////////////////////////////
void GoldenPatternBase::finalise(unsigned int procIndx) {
  for(auto& result : getResults()[procIndx]) {
    result.finalise();
  }
}
