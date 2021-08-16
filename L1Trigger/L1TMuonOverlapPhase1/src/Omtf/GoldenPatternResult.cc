#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPatternResult.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFConfiguration.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <ostream>
#include <iomanip>
#include <cmath>

////////////////////////////////////////////
////////////////////////////////////////////

////////////////////////////////////////////
////////////////////////////////////////////
GoldenPatternResult::GoldenPatternResult(const OMTFConfiguration* omtfConfig)
    : finalise([this]() { finalise0(); }), omtfConfig(omtfConfig) {
  if (omtfConfig)
    init(omtfConfig);
}

////////////////////////////////////////////
////////////////////////////////////////////
/*void GoldenPatternResult::configure(const OMTFConfiguration * omtfConfig) {
  myOmtfConfig = omtfConfig;
  assert(myOmtfConfig != 0);
  reset();
}*/
////////////////////////////////////////////
////////////////////////////////////////////

void GoldenPatternResult::set(int refLayer_, int phi, int eta, int refHitPhi) {
  if (isValid() && this->refLayer != refLayer_) {
    std::cout << __FUNCTION__ << " " << __LINE__ << " this->refLayer " << this->refLayer << " refLayer_ " << refLayer_
              << std::endl;
  }
  assert(!isValid() || this->refLayer == refLayer_);

  this->refLayer = refLayer_;
  this->phi = phi;
  this->eta = eta;
  this->refHitPhi = refHitPhi;
}

void GoldenPatternResult::setStubResult(float pdfVal, bool valid, int pdfBin, int layer, MuonStubPtr stub) {
  if (valid) {
    //pdfSum += pdfVal;
    //firedLayerBits.set(layer);
    firedLayerBits |= (1 << layer);
  }
  stubResults[layer] = StubResult(pdfVal, valid, pdfBin, layer, stub);

  //stub result is added evevn thought it is not valid since this might be needed for debugging or optimization
}

void GoldenPatternResult::setStubResult(int layer, StubResult& stubResult) {
  if (stubResult.getValid()) {
    //pdfSum += pdfVal;
    //firedLayerBits.set(layer);
    firedLayerBits |= (1 << layer);
  }
  stubResults[layer] = stubResult;

  //stub result is added evevn thought it is not valid since this might be needed for debugging or optimization
}

////////////////////////////////////////////
////////////////////////////////////////////
void GoldenPatternResult::init(const OMTFConfiguration* omtfConfig) {
  this->omtfConfig = omtfConfig;

  finalizeFunction = this->omtfConfig->getGoldenPatternResultFinalizeFunction();

  if (finalizeFunction == 1)
    finalise = [this]() { finalise1(); };
  else if (finalizeFunction == 2)
    finalise = [this]() { finalise2(); };
  else if (finalizeFunction == 3)
    finalise = [this]() { finalise3(); };
  else if (finalizeFunction == 5)
    finalise = [this]() { finalise5(); };
  else if (finalizeFunction == 6)
    finalise = [this]() { finalise6(); };
  else if (finalizeFunction == 7)
    finalise = [this]() { finalise7(); };
  else if (finalizeFunction == 8)
    finalise = [this]() { finalise8(); };
  else if (finalizeFunction == 9)
    finalise = [this]() { finalise9(); };
  else
    finalise = [this]() { finalise0(); };

  stubResults.assign(omtfConfig->nLayers(), StubResult());
  reset();
}

void GoldenPatternResult::reset() {
  for (auto& stubResult : stubResults) {
    stubResult.reset();
  }
  valid = false;
  refLayer = -1;
  phi = 0;
  eta = 0;
  pdfSum = 0;
  firedLayerCnt = 0;
  firedLayerBits = 0;
  refHitPhi = 0;
  gpProbability1 = 0;
  gpProbability2 = 0;
}

////////////////////////////////////////////
////////////////////////////////////////////
//default version
void GoldenPatternResult::finalise0() {
  for (unsigned int iLogicLayer = 0; iLogicLayer < stubResults.size(); ++iLogicLayer) {
    unsigned int connectedLayer = omtfConfig->getLogicToLogic().at(iLogicLayer);
    //here we require that in case of the DT layers, both phi and phiB is fired
    if (firedLayerBits & (1 << connectedLayer)) {
      if (firedLayerBits & (1 << iLogicLayer)) {
        //now in the GoldenPattern::process1Layer1RefLayer the pdf bin 0 is returned when the layer is not fired, so this is 'if' is to assured that this pdf val is not added here
        pdfSum += stubResults[iLogicLayer].getPdfVal();

        if (omtfConfig->fwVersion() <= 4) {
          if (!omtfConfig->getBendingLayers().count(iLogicLayer))
            //in DT case, the phi and phiB layers are threaded as one, so the firedLayerCnt is increased only for the phi layer
            firedLayerCnt++;
        } else
          firedLayerCnt++;
      }
    } else {
      firedLayerBits &= ~(1 << iLogicLayer);
    }
  }

  valid = true;
  //by default result becomes valid here, but can be overwritten later
}

////////////////////////////////////////////
////////////////////////////////////////////
//for the algo version with thresholds
void GoldenPatternResult::finalise1() {
  //cout<<__FUNCTION__<<":"<<__LINE__<<endl;
  for (unsigned int iLogicLayer = 0; iLogicLayer < stubResults.size(); ++iLogicLayer) {
    //in this version we do not require that both phi and phiB is fired (non-zero), we thread them just independent
    //watch out that then the number of fired layers is bigger, and the cut on the minimal number of fired layers does not work in the same way as when the dt chamber is counted as one layer
    //TODO check if it affects performance
    pdfSum += stubResults[iLogicLayer].getPdfVal();
    firedLayerCnt += ((firedLayerBits & (1 << iLogicLayer)) != 0);
  }

  valid = true;
  //by default result becomes valid here, but can be overwritten later
}

////////////////////////////////////////////
////////////////////////////////////////////
//multiplication of PDF values instead of sum
void GoldenPatternResult::finalise2() {
  pdfSum = 1.;
  firedLayerCnt = 0;
  for (unsigned int iLogicLayer = 0; iLogicLayer < stubResults.size(); ++iLogicLayer) {
    unsigned int connectedLayer = omtfConfig->getLogicToLogic().at(iLogicLayer);
    //here we require that in case of the DT layers, both phi and phiB is fired
    if (firedLayerBits & (1 << connectedLayer)) {
      if (firedLayerBits &
          (1
           << iLogicLayer)) {  //now in the GoldenPattern::process1Layer1RefLayer the pdf bin 0 is returned when the layer is not fired, so this is 'if' is to assured that this pdf val is not added here
        pdfSum *= stubResults[iLogicLayer].getPdfVal();
        if (!omtfConfig->getBendingLayers().count(
                iLogicLayer))  //in DT case, the phi and phiB layers are threaded as one, so the firedLayerCnt is increased only for the phi layer
          firedLayerCnt++;
      }
    } else {
      firedLayerBits &= ~(1 << iLogicLayer);
    }
  }

  if (firedLayerCnt < 3)
    pdfSum = 0;

  valid = true;
  //by default result becomes valid here, but can be overwritten later
}

////////////////////////////////////////////
////////////////////////////////////////////
//for patterns generation
void GoldenPatternResult::finalise3() {
  firedLayerCnt = 0;
  for (unsigned int iLogicLayer = 0; iLogicLayer < stubResults.size(); ++iLogicLayer) {
    //in this version we do not require that both phi and phiB is fired (non-zero), we thread them just independent
    //watch out that then the number of fired layers is bigger, and the cut on the minimal number of fired layers dies not work in the same way as when the dt chamber is counted as one layer
    //TODO check if it affects performance
    pdfSum += stubResults[iLogicLayer].getPdfVal();

    if (stubResults[iLogicLayer].getMuonStub())
      firedLayerCnt++;
  }

  valid = true;
  //by default result becomes valid here, but can be overwritten later
}

void GoldenPatternResult::finalise5() {
  for (unsigned int iLogicLayer = 0; iLogicLayer < stubResults.size(); ++iLogicLayer) {
    unsigned int connectedLayer = omtfConfig->getLogicToLogic().at(iLogicLayer);

    if (omtfConfig->isBendingLayer(iLogicLayer)) {  //the DT phiB layer is counted only when the phi layer is fired
      if ((firedLayerBits & (1 << iLogicLayer)) && (firedLayerBits & (1 << connectedLayer))) {
        pdfSum += stubResults[iLogicLayer].getPdfVal();
        firedLayerCnt++;
      } else {
        firedLayerBits &= ~(1 << iLogicLayer);
        stubResults[iLogicLayer].setValid(false);
        //in principle the stun should be also removed from the stubResults[iLogicLayer], on the other hand ini this way can be used e.g. for debug
      }
    } else if (firedLayerBits & (1 << iLogicLayer)) {
      pdfSum += stubResults[iLogicLayer].getPdfVal();
      firedLayerCnt++;
    }
  }

  valid = true;
  //by default result becomes valid here, but can be overwritten later
}

void GoldenPatternResult::finalise6() {
  for (unsigned int iLogicLayer = 0; iLogicLayer < stubResults.size(); ++iLogicLayer) {
    unsigned int connectedLayer = omtfConfig->getLogicToLogic().at(iLogicLayer);

    if (omtfConfig->isBendingLayer(iLogicLayer)) {  //the DT phiB layer is counted only when the phi layer is fired
      if ((firedLayerBits & (1 << iLogicLayer)) && (firedLayerBits & (1 << connectedLayer)) &&
          (stubResults[iLogicLayer].getMuonStub()->qualityHw >= 4)) {
        pdfSum += stubResults[iLogicLayer].getPdfVal();
        firedLayerCnt++;
      } else {
        firedLayerBits &= ~(1 << iLogicLayer);
        stubResults[iLogicLayer].setValid(false);
        //in principle the stun should be also removed from the stubResults[iLogicLayer], on the other hand ini this way can be used e.g. for debug
      }
    } else if (firedLayerBits & (1 << iLogicLayer)) {
      pdfSum += stubResults[iLogicLayer].getPdfVal();
      firedLayerCnt++;
    }
  }

  valid = true;
  //by default result becomes valid here, but can be overwritten later
}

void GoldenPatternResult::finalise7() {
  for (unsigned int iLogicLayer = 0; iLogicLayer < stubResults.size(); ++iLogicLayer) {
    pdfSum += stubResults[iLogicLayer].getPdfVal();
    if (firedLayerBits & (1 << iLogicLayer)) {
      firedLayerCnt++;
    }
  }

  valid = true;
  //by default result becomes valid here, but can be overwritten later
}

void GoldenPatternResult::finalise8() {
  for (unsigned int iLogicLayer = 0; iLogicLayer < stubResults.size(); ++iLogicLayer) {
    pdfSum += stubResults[iLogicLayer].getPdfVal();  //pdfSum is counted always

    unsigned int connectedLayer = omtfConfig->getLogicToLogic().at(iLogicLayer);
    if (omtfConfig->isBendingLayer(iLogicLayer)) {  //the DT phiB layer is counted only when the phi layer is fired
      if ((firedLayerBits & (1 << iLogicLayer)) && (firedLayerBits & (1 << connectedLayer))) {
        // && (stubResults[iLogicLayer].getMuonStub()->qualityHw >= 4) this is not needed, as the rejecting the low quality phiB hits is on the input of the algorithm
        firedLayerCnt++;
      } else {
        firedLayerBits &= ~(1 << iLogicLayer);
        stubResults[iLogicLayer].setValid(false);
        //in principle the stub should be also removed from the stubResults[iLogicLayer], on the other hand in this way can be used e.g. for debug
      }
    } else if (firedLayerBits & (1 << iLogicLayer)) {
      firedLayerCnt++;
    }
  }

  valid = true;
  //by default result becomes valid here, but can be overwritten later
}

void GoldenPatternResult::finalise9() {
  for (unsigned int iLogicLayer = 0; iLogicLayer < stubResults.size(); ++iLogicLayer) {
    unsigned int connectedLayer = omtfConfig->getLogicToLogic().at(iLogicLayer);

    if (omtfConfig->isBendingLayer(iLogicLayer)) {  //the DT phiB layer is counted only when the phi layer is fired
      if (firedLayerBits & (1 << iLogicLayer)) {
        if (firedLayerBits & (1 << connectedLayer)) {
          firedLayerCnt++;
          pdfSum += stubResults[iLogicLayer].getPdfVal();
        } else {
          firedLayerBits &= ~(1 << iLogicLayer);
          stubResults[iLogicLayer].setValid(false);
          //if(stubResults[iLogicLayer].getPdfVal() == 0) pdfSum -= 64;; //there was hit, but it did not fire to the pdf - this is not possible here, since the banding layer if fired here
          //so in this case simply:
          //pdfSum += 0;
        }
      } else {
        //banding layer fired, but not fits to the pdf, N.B works only with the patterns having "no hit value" and with noHitValueInPdf = True
        if (stubResults[iLogicLayer].getPdfVal() == 0)
          pdfSum -= 32;
        else
          pdfSum += stubResults[iLogicLayer].getPdfVal();  //banding layer not fired at all
      }
    } else {
      if (iLogicLayer < 10 && stubResults[iLogicLayer].getPdfVal() == 0)
        pdfSum -= 32;
      else
        pdfSum += stubResults[iLogicLayer].getPdfVal();
      if (firedLayerBits & (1 << iLogicLayer)) {  //pdfSum is counted always
        firedLayerCnt++;
      }
    }
  }

  valid = true;
  //by default result becomes valid here, but can be overwritten later
}

/*void GoldenPatternResult::finalise2() {
  pdfSum = 1.;
  for(unsigned int iLogicLayer=0; iLogicLayer < pdfValues.size(); ++iLogicLayer) {
    //in this version we do not require that both phi and phiB is fired (non-zero), we thread them just independent
    //TODO check if it affects performance
    bool layerFired = ( (firedLayerBits & (1<<iLogicLayer)) != 0 );
    if(layerFired) {
      pdfSum *= pdfValues[iLogicLayer];
      firedLayerCnt++ ;
    }
  }
  if(firedLayerCnt < 3)
    pdfSum = 0;
  valid = true;
  //by default result becomes valid here, but can be overwritten later
}*/

////////////////////////////////////////////
////////////////////////////////////////////
std::ostream& operator<<(std::ostream& out, const GoldenPatternResult& gpResult) {
  unsigned int refLayerLogicNum = gpResult.omtfConfig->getRefToLogicNumber()[gpResult.getRefLayer()];

  unsigned int sumOverFiredLayers = 0;
  for (unsigned int iLogicLayer = 0; iLogicLayer < gpResult.stubResults.size(); ++iLogicLayer) {
    out << " layer: " << std::setw(2) << iLogicLayer << " hit: ";
    if (gpResult.stubResults[iLogicLayer].getMuonStub()) {
      out << std::setw(4)
          << (gpResult.omtfConfig->isBendingLayer(iLogicLayer)
                  ? gpResult.stubResults[iLogicLayer].getMuonStub()->phiBHw
                  : gpResult.stubResults[iLogicLayer].getMuonStub()->phiHw);

      out << " pdfBin: " << std::setw(4) << gpResult.stubResults[iLogicLayer].getPdfBin() << " pdfVal: " << std::setw(3)
          << gpResult.stubResults[iLogicLayer].getPdfVal() << " fired " << gpResult.isLayerFired(iLogicLayer)
          << (iLogicLayer == refLayerLogicNum ? " <<< refLayer" : "");

      if (gpResult.isLayerFired(iLogicLayer))
        sumOverFiredLayers += gpResult.stubResults[iLogicLayer].getPdfVal();
    } else if (gpResult.stubResults[iLogicLayer].getPdfVal()) {
      out << "                  pdfVal: " << std::setw(3) << gpResult.stubResults[iLogicLayer].getPdfVal();
    }
    out << std::endl;
  }

  out << "  refLayer: ";
  out << gpResult.getRefLayer() << "\t";

  out << " Sum over layers: ";
  out << gpResult.getPdfSum() << "\t";

  out << " sumOverFiredLayers: ";
  out << sumOverFiredLayers << "\t";

  out << " Number of hits: ";
  out << gpResult.getFiredLayerCnt() << "\t";

  out << " GpProbability1: ";
  out << gpResult.getGpProbability1() << "\t";

  out << " GpProbability2: ";
  out << gpResult.getGpProbability2() << "\t";

  out << std::endl;

  return out;
}
////////////////////////////////////////////
////////////////////////////////////////////
