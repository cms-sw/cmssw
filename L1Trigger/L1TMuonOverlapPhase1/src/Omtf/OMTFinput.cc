#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFConfiguration.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFinput.h"

#include <iomanip>

///////////////////////////////////////////////////
///////////////////////////////////////////////////
const int inputsPerLayer = 14;
OMTFinput::OMTFinput(const OMTFConfiguration* omtfConfig) : MuonStubsInput(omtfConfig) {
  myOmtfConfig = omtfConfig;
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
  /*  assert(iLayer < muonStubsInLayers.size());
  assert(iInput < muonStubsInLayers[iLayer].size());*/
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

///////////////////////////////////////////////////
///////////////////////////////////////////////////
std::bitset<128> OMTFinput::getRefHits(unsigned int iProcessor) const {
  std::bitset<128> refHits;

  unsigned int iRefHit = 0;
  for (auto iRefHitDef : myOmtfConfig->getRefHitsDefs()[iProcessor]) {
    int iPhi = getPhiHw(myOmtfConfig->getRefToLogicNumber()[iRefHitDef.iRefLayer], iRefHitDef.iInput);
    if (iPhi < (int)myOmtfConfig->nPhiBins()) {
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
