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
    //pdfSum and firedLayerBits is calculated in finaliseX()
    firedLayerBits |= (1 << layer);
  }
  stubResults[layer] = StubResult(pdfVal, valid, pdfBin, layer, stub);

  //stub result is added even thought it is not valid since this might be needed for debugging or optimization
}

void GoldenPatternResult::setStubResult(int layer, StubResult& stubResult) {
  if (stubResult.getValid()) {
    //pdfSum and firedLayerBits is calculated in finaliseX()
    firedLayerBits |= (1 << layer);
  }
  stubResults[layer] = stubResult;

  //stub result is added even thought it is not valid since this might be needed for debugging or optimization
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
  else if (finalizeFunction == 10)
    finalise = [this]() { finalise10(); };
  else if (finalizeFunction == 11)
    finalise = [this]() { finalise11(); };
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
  pdfSumUnconstr = 0;
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
      //now in the GoldenPattern::process1Layer1RefLayer the pdf bin 0 is returned when the layer is not fired, so this is 'if' is to assured that this pdf val is not added here
      if (firedLayerBits & (1 << iLogicLayer)) {
        pdfSum *= stubResults[iLogicLayer].getPdfVal();
        //in DT case, the phi and phiB layers are threaded as one, so the firedLayerCnt is increased only for the phi layer
        if (!omtfConfig->getBendingLayers().count(iLogicLayer))
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
          //there was hit, but it did not fit to the pdf - this is not possible here, since the bending layer is fired here
          //therefore there is no sense to apply the penalty when the stubResults[iLogicLayer].getPdfVal() == 0
          //so in this case simply pdfSum += 0;
        }
      } else {
        if (stubResults[iLogicLayer].getPdfVal() == 0)
          //there is a hit, but does not fit to the pdf (therefore in firedLayerBits is 0, but getPdfVal() is not 0), so apply the penalty (-32)
          //N.B it is possible only with the patterns having "no hit value" and with noHitValueInPdf = True
          pdfSum -= 32;  // penaly
        else
          pdfSum += stubResults[iLogicLayer].getPdfVal();  //bending layer not fired at all
      }
    } else {
      if (iLogicLayer < 10 && stubResults[iLogicLayer].getPdfVal() == 0)
        pdfSum -= 32;  // penaly
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

void GoldenPatternResult::finalise10() {
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
          //there is no sense to apply the penalty in this case,
          //because as the layer is fired, the stubResults[iLogicLayer].getPdfVal() cannot be 0
          //so in this case simply pdfSum += 0;
        }
      } else {
        //the penalty is not applied here when the phiB does not fit to the pdf
        //because when extrapolation from the ref layer using the phiB is applied
        //it "normal" for the displaced muons to not fit to the pdf
        pdfSum += stubResults[iLogicLayer].getPdfVal();
      }
    } else {
      if (iLogicLayer < 10 && stubResults[iLogicLayer].getPdfVal() == 0)
        pdfSum -= 32;  // penaly
      else
        pdfSum += stubResults[iLogicLayer].getPdfVal();
      if (firedLayerBits & (1 << iLogicLayer)) {  //pdfSum is counted always
        firedLayerCnt++;
      }
    }
  }

  if ((omtfConfig->usePhiBExtrapolationMB1() && refLayer == 0) ||
      (omtfConfig->usePhiBExtrapolationMB2() && refLayer == 2)) {
    auto refLayerLogicNumber = omtfConfig->getRefToLogicNumber()[refLayer];
    //Unconstrained pt is obtained by not including the pdfValue from the phiB of the refHit
    //TODO get logic layer from connectedLayer
    pdfSumUnconstr = pdfSum - stubResults[refLayerLogicNumber + 1].getPdfVal();
    //here there is an issue with the firedLayerBits and quality assignment:
    //in case if the displaced muon the phiB layer of the ref hit might not be fired (pdfVal might be 0)
    //which in principle has no sense, because by the displaced algorithm construction it is fired
    //an effect of that is that some fraction of displaced muons get the quality 8 assigned
    //the efficiency difference between quality 8 and 12 seems to be at a level of 1-2%
    //but in the uGT menu e.g. the L1_DoubleMu0_Upt6_IP_Min1_Upt4 uses quality >= 0, so should be OK

    //hard cut - the phiB of the refHit must fit to the pdfS
    //but this cut has sometimes side effect: there can be a muon which has has pdfSum = 0 for every pattern,
    //then in the OMTFSorter<GoldenPatternType>::sortRefHitResults the first pattern that has FiredLayerCnt >= 3 is chosen
    //and not the one with highest pdfSum as it should be
    //TODO what should be done is to set the pt of such a muons to 0, but after the sorter.
    //Or maybe not - if the pt is 0, then the muon is not valid. So the displaced muon will be lost.
    //So the way it is done now actually is good. Such a muon will have some low constrained pt probably.
    //what can be done is to assign to it the hwPt = 1 , but not 0
    //TODO modify this condition to use the firedLayerBits and not getPdfVal
    //as it would be much easier for the firmware
    if (stubResults[refLayerLogicNumber + 1].getPdfVal() == 0)
      pdfSum = 0;
  } else
    pdfSumUnconstr = 0;

  valid = true;
  //by default result becomes valid here, but can be overwritten later
}

//  the same as finalise10 but without:
//if (stubResults[refLayerLogicNumber + 1].getPdfVal() == 0)
//  pdfSum = 0;
void GoldenPatternResult::finalise11() {
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
          //there is no sense to apply the penalty in this case,
          //because as the layer is fired, the stubResults[iLogicLayer].getPdfVal() cannot be 0
          //so in this case simply pdfSum += 0;
        }
      } else {
        //bending layer fired, but not fits to the pdf, N.B works only with the patterns having "no hit value" and with noHitValueInPdf = True
        if (stubResults[iLogicLayer].getPdfVal() == 0) {
          //high penalty, we set the pdf value in the  stubResults[iLogicLayer], so that this penalty is removed from pdfSumUnconstr
          pdfSum -= 63;
          stubResults[iLogicLayer].setPdfVal(-63);
        } else
          pdfSum += stubResults[iLogicLayer].getPdfVal();  //bending layer not fired at all
      }
    } else {
      if (iLogicLayer < 10 && stubResults[iLogicLayer].getPdfVal() == 0)
        pdfSum -= 32;  // penaly
      else
        pdfSum += stubResults[iLogicLayer].getPdfVal();
      if (firedLayerBits & (1 << iLogicLayer)) {  //pdfSum is counted always
        firedLayerCnt++;
      }
    }
  }

  if ((omtfConfig->usePhiBExtrapolationMB1() && refLayer == 0) ||
      (omtfConfig->usePhiBExtrapolationMB2() && refLayer == 2)) {
    auto refLayerLogicNumber = omtfConfig->getRefToLogicNumber()[refLayer];
    //Unconstrained pt is obtained by not including the pdfValue from the phiB of the refHit
    //TODO get logic layer from connectedLayer
    pdfSumUnconstr = pdfSum - stubResults[refLayerLogicNumber + 1].getPdfVal();

  } else
    pdfSumUnconstr = 0;

  valid = true;
  //by default result becomes valid here, but can be overwritten later
}

////////////////////////////////////////////
////////////////////////////////////////////
std::ostream& operator<<(std::ostream& out, const GoldenPatternResult& gpResult) {
  if (gpResult.omtfConfig == nullptr) {
    out << "empty GoldenPatternResult" << std::endl;
    return out;
  }
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

  out << " Sum over layers unconstr: ";
  out << gpResult.getPdfSumUnconstr() << "\t";

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
