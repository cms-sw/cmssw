#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFConfiguration.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFinput.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iomanip>

///////////////////////////////////////////////////
///////////////////////////////////////////////////
OMTFinput::OMTFinput(const OMTFConfiguration* omtfConfig) : MuonStubsInput(omtfConfig) {
  myOmtfConfig = omtfConfig;
  const int inputsPerLayer = myOmtfConfig->nInputs();
  muonStubsInLayers.assign(omtfConfig->nLayers(), std::vector<MuonStubPtr>(inputsPerLayer));
  //nullptrs are assigned here for every input
}
///////////////////////////////////////////////////
///////////////////////////////////////////////////

//TODO remove and leave only the MuonStubsInput::getPhiHw
int OMTFinput::getPhiHw(unsigned int iLayer, unsigned int iInput) const {
  /*  assert(iLayer < muonStubsInLayers.size());
  assert(iInput < muonStubsInLayers[iLayer].size());*/
  if (this->myOmtfConfig->isBendingLayer(iLayer)) {
    MuonStubPtr stub = getMuonStub(iLayer - 1, iInput);
    if (stub)
      return stub->phiBHw;
  }

  MuonStubPtr stub = getMuonStub(iLayer, iInput);
  if (stub)
    return stub->phiHw;

  return myOmtfConfig->nPhiBins();
}

const int OMTFinput::getHitEta(unsigned int iLayer, unsigned int iInput) const {
  if (this->myOmtfConfig->isBendingLayer(iLayer)) {
    MuonStubPtr stub = getMuonStub(iLayer - 1, iInput);
    if (stub)
      return stub->etaHw;
  }

  MuonStubPtr stub = getMuonStub(iLayer, iInput);
  if (stub)
    return stub->etaHw;

  return myOmtfConfig->nPhiBins();
}

const int OMTFinput::getHitQual(unsigned int iLayer, unsigned int iInput) const {
  /*  assert(iLayer < muonStubsInLayers.size());
  assert(iInput < muonStubsInLayers[iLayer].size());*/
  if (this->myOmtfConfig->isBendingLayer(iLayer)) {
    MuonStubPtr stub = getMuonStub(iLayer - 1, iInput);
    if (stub)
      return stub->qualityHw;
  }

  MuonStubPtr stub = getMuonStub(iLayer, iInput);
  if (stub)
    return stub->qualityHw;

  return myOmtfConfig->nPhiBins();
}

///////////////////////////////////////////////////
///////////////////////////////////////////////////
boost::dynamic_bitset<> OMTFinput::getRefHits(unsigned int iProcessor) const {
  boost::dynamic_bitset<> refHits(myOmtfConfig->nRefHits());

  unsigned int iRefHit = 0;
  for (auto iRefHitDef : myOmtfConfig->getRefHitsDefs()[iProcessor]) {
    auto refHitLogicLayer = myOmtfConfig->getRefToLogicNumber()[iRefHitDef.iRefLayer];

    int iPhi = getPhiHw(refHitLogicLayer, iRefHitDef.iInput);
    if (iPhi < (int)myOmtfConfig->nPhiBins()) {
      //TODO use a constant defined somewhere instead of 6
      if (refHitLogicLayer >= 6 ||
          getMuonStub(refHitLogicLayer, iRefHitDef.iInput)->qualityHw >= myOmtfConfig->getDtRefHitMinQuality())
        refHits.set(iRefHit, iRefHitDef.fitsRange(iPhi));
    }
    iRefHit++;
  }

  return refHits;
}

///////////////////////////////////////////////////
///////////////////////////////////////////////////
std::ostream& operator<<(std::ostream& out, const OMTFinput& aInput) {
  for (unsigned int iLogicLayer = 0; iLogicLayer < aInput.muonStubsInLayers.size(); ++iLogicLayer) {
    out << "Logic layer: " << std::setw(2) << iLogicLayer << " Hits: ";
    for (unsigned int iHit = 0; iHit < aInput.muonStubsInLayers[iLogicLayer].size(); ++iHit) {
      //out<<aInput.muonStubsInLayers[iLogicLayer][iHit]<<"\t";
      int phi = aInput.getPhiHw(iLogicLayer, iHit);
      if (phi == 5400)
        out << std::setw(4) << "...."
            << " ";
      else
        out << std::setw(4) << phi << " ";
      //TODO print other value?
    }
    out << std::endl;
  }
  return out;
}
///////////////////////////////////////////////////
///////////////////////////////////////////////////
