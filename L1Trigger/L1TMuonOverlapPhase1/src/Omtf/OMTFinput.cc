#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFConfiguration.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFinput.h"

#include <iomanip>

///////////////////////////////////////////////////
///////////////////////////////////////////////////
const int inputsPerLayer = 14;
OMTFinput::OMTFinput(const OMTFConfiguration* omtfConfig) : MuonStubsInput(omtfConfig) {
  myOmtfConfig = omtfConfig;
  //muonStubsInLayers.assign(omtfConfig->nLayers(), std::vector<MuonStub>(inputsPerLayer, MuonStub(myOmtfConfig->nPhiBins(), myOmtfConfig->nPhiBins())) );
  muonStubsInLayers.assign(
      omtfConfig->nLayers(),
      std::vector<MuonStubPtr>(
          inputsPerLayer));  //, MuonStub(myOmtfConfig->nPhiBins(), myOmtfConfig->nPhiBins()) TODO do we want to create the MuonStubs for every input???
  //clear();
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
/*bool OMTFinput::addLayerHit(unsigned int iLayer,
			    unsigned int iInput,
			    int iPhi, int iEta, bool allowOverwrite){

  bool overwrite = false;
  assert(iLayer<myOmtfConfig->nLayers());
  assert(iInput<14); //FIXME define parameters for this 14

  if(iPhi>=(int)myOmtfConfig->nPhiBins())
    return true;

  if(allowOverwrite && muonStubsInLayers[iLayer][iInput].phiHw == iPhi && muonStubsInLayers[iLayer][iInput].etaHw == iEta)
    return true;

  if(muonStubsInLayers[iLayer][iInput].phiHw != (int)myOmtfConfig->nPhiBins()) ++iInput;
  if(muonStubsInLayers[iLayer][iInput].phiHw != (int)myOmtfConfig->nPhiBins()) overwrite = true;
  
  if(iInput >= 14)
    return true;
  
  muonStubsInLayers[iLayer][iInput].phiHw = iPhi;
  muonStubsInLayers[iLayer][iInput].etaHw = iEta;

  return overwrite;				      
}*/

///////////////////////////////////////////////////
///////////////////////////////////////////////////
/*void OMTFinput::readData(XMLConfigReader *aReader,
			 unsigned int iEvent,
			 unsigned int iProcessor){

  measurementsPhi = aReader->readEvent(iEvent, iProcessor);
  measurementsEta = aReader->readEvent(iEvent, iProcessor, true);
  
}*/
///////////////////////////////////////////////////
///////////////////////////////////////////////////
/*void OMTFinput::mergeData(const OMTFinput *aInput){

  for(unsigned int iLayer=0;iLayer<myOmtfConfig->nLayers();++iLayer){
    const OMTFinput::vector1D & aPhiVec = aInput->getLayerData(iLayer,false);
    const OMTFinput::vector1D & aEtaVec = aInput->getLayerData(iLayer,true);
    if(aPhiVec.empty()) continue;

    OMTFinput::vector1D layerData = getLayerData(iLayer, false);
    for(unsigned int iInput=0;iInput<14;++iInput){      
      addLayerHit(iLayer,iInput,aPhiVec[iInput],aEtaVec[iInput]);
    }
  }
}*/
///////////////////////////////////////////////////
///////////////////////////////////////////////////
/*void OMTFinput::clear() {
  for(auto& muonStubsInLayer : muonStubsInLayers) {
    for(auto& muonStub : muonStubsInLayer) {
      muonStub = MuonStub();
      muonStub.phiHw = myOmtfConfig->nPhiBins();
      muonStub.phiBHw = myOmtfConfig->nPhiBins();
    }
  }
}*/
///////////////////////////////////////////////////
///////////////////////////////////////////////////
/*void  OMTFinput::shiftMyPhi(int phiShift){

  int lowScaleEnd = std::pow(2,myOmtfConfig->nPhiBits()-1);
  int highScaleEnd = lowScaleEnd-1;

for(unsigned int iLogicLayer=0;iLogicLayer<measurementsPhi.size();++iLogicLayer){
    for(unsigned int iHit=0;iHit<measurementsPhi[iLogicLayer].size();++iHit){
      if(!myOmtfConfig->getBendingLayers().count(iLogicLayer) &&
	 measurementsPhi[iLogicLayer][iHit]<(int)myOmtfConfig->nPhiBins()){
	if(measurementsPhi[iLogicLayer][iHit]<0) measurementsPhi[iLogicLayer][iHit]+=myOmtfConfig->nPhiBins();
	measurementsPhi[iLogicLayer][iHit]-=phiShift;
	if(measurementsPhi[iLogicLayer][iHit]<0) measurementsPhi[iLogicLayer][iHit]+=myOmtfConfig->nPhiBins();
	measurementsPhi[iLogicLayer][iHit]+=-lowScaleEnd;
	if(measurementsPhi[iLogicLayer][iHit]<-lowScaleEnd ||
	   measurementsPhi[iLogicLayer][iHit]>highScaleEnd) measurementsPhi[iLogicLayer][iHit] = (int)myOmtfConfig->nPhiBins();	   
      }
    }
  }
}*/
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
