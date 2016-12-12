#include <iostream>
#include <algorithm>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

#include "L1Trigger/L1TMuonOverlap/interface/OMTFConfiguration.h"

///////////////////////////////////////////////
///////////////////////////////////////////////
RefHitDef::RefHitDef(unsigned int aInput, 
		     int aPhiMin, int aPhiMax,
		     unsigned int aRegion,
		     unsigned int aRefLayer):
  iInput(aInput),
  iRegion(aRegion),
  iRefLayer(aRefLayer),
  range(std::pair<int,int>(aPhiMin,aPhiMax)){}
///////////////////////////////////////////////
///////////////////////////////////////////////
bool RefHitDef::fitsRange(int iPhi) const{

  return iPhi>=range.first && 
         iPhi<range.second;

}
///////////////////////////////////////////////
///////////////////////////////////////////////
std::ostream & operator << (std::ostream &out, const  RefHitDef & aRefHitDef){


  out<<"iRefLayer: "<<aRefHitDef.iRefLayer
     <<" iInput: "<<aRefHitDef.iInput
     <<" iRegion: "<<aRefHitDef.iRegion
     <<" range: ("<<aRefHitDef.range.first
     <<", "<<aRefHitDef.range.second<<std::endl;

  return out;
}
///////////////////////////////////////////////
///////////////////////////////////////////////
void OMTFConfiguration::initCounterMatrices(){
  
  ///Vector of all inputs
  std::vector<int> aLayer1D(nInputs(),0);

  ///Vector of all layers 
  vector2D aLayer2D;
  aLayer2D.assign(nLayers(),aLayer1D);

  ///Vector of all logic cones
  vector3D aLayer3D;
  aLayer3D.assign(nLogicRegions(),aLayer2D);

  ///Vector of all processors
  measurements4D.assign(nProcessors(),aLayer3D);
  measurements4Dref.assign(nProcessors(),aLayer3D);
}
///////////////////////////////////////////////
///////////////////////////////////////////////
void OMTFConfiguration::configure(const L1TMuonOverlapParams *omtfParams){


  rawParams = *omtfParams;
 
  ///Set chamber sectors connections to logic processros.
  barrelMin.resize(nProcessors());
  endcap10DegMin.resize(nProcessors());
  endcap20DegMin.resize(nProcessors());

  barrelMax.resize(nProcessors());
  endcap10DegMax.resize(nProcessors());
  endcap20DegMax.resize(nProcessors());

  const std::vector<int> *connectedSectorsStartVec =  omtfParams->connectedSectorsStart();
  const std::vector<int> *connectedSectorsEndVec =  omtfParams->connectedSectorsEnd();

  std::copy(connectedSectorsStartVec->begin(), connectedSectorsStartVec->begin()+6, barrelMin.begin());  
  std::copy(connectedSectorsStartVec->begin()+6, connectedSectorsStartVec->begin()+12, endcap10DegMin.begin());
  std::copy(connectedSectorsStartVec->begin()+12, connectedSectorsStartVec->end(), endcap20DegMin.begin());

  std::copy(connectedSectorsEndVec->begin(), connectedSectorsEndVec->begin()+6, barrelMax.begin());
  std::copy(connectedSectorsEndVec->begin()+6, connectedSectorsEndVec->begin()+12, endcap10DegMax.begin());
  std::copy(connectedSectorsEndVec->begin()+12, connectedSectorsEndVec->end(), endcap20DegMax.begin());

  ///Set connections tables
  const std::vector<L1TMuonOverlapParams::LayerMapNode> *layerMap = omtfParams->layerMap();

  for(unsigned int iLayer=0;iLayer<nLayers();++iLayer){
    L1TMuonOverlapParams::LayerMapNode aNode = layerMap->at(iLayer);    
    hwToLogicLayer[aNode.hwNumber] = aNode.logicNumber;
    logicToHwLayer[aNode.logicNumber] = aNode.hwNumber;
    logicToLogic[aNode.logicNumber] = aNode.connectedToLayer;    
    if(aNode.bendingLayer) bendingLayers.insert(aNode.logicNumber);        
  }
  /////
  refToLogicNumber.resize(nRefLayers());
  
  const std::vector<L1TMuonOverlapParams::RefLayerMapNode> *refLayerMap = omtfParams->refLayerMap();
  for(unsigned int iRefLayer=0;iRefLayer<nRefLayers();++iRefLayer){
    L1TMuonOverlapParams::RefLayerMapNode aNode = refLayerMap->at(iRefLayer);    
    refToLogicNumber[aNode.refLayer] = aNode.logicNumber;
  }
  /////
  std::vector<int> vector1D(nRefLayers(),nPhiBins());
  processorPhiVsRefLayer.assign(nProcessors(),vector1D);

  ///connections tables for each processor each logic cone
  ///Vector of all layers
  vector1D_pair aLayer1D(nLayers());
  ///Vector of all logic cones
  vector2D_pair aLayer2D;
  aLayer2D.assign(nLogicRegions(),aLayer1D);
  ///Vector of all processors
  connections.assign(nProcessors(),aLayer2D);

  ///Starting phis of each region
  ///Vector of all regions in one processor
  std::vector<std::pair<int,int> > aRefHit1D(nLogicRegions(),std::pair<int,int>(9999,9999));
  ///Vector of all reflayers
  std::vector<std::vector<std::pair<int,int> > > aRefHit2D;
  aRefHit2D.assign(nRefLayers(),aRefHit1D);
  ///Vector of all inputs
  regionPhisVsRefLayerVsInput.assign(nInputs(),aRefHit2D);

  //Vector of ref hit definitions
  std::vector<RefHitDef> aRefHitsDefs(nRefHits());
  ///Vector of all processros
  refHitsDefs.assign(nProcessors(),aRefHitsDefs);

  const std::vector<int> *phiStartMap =  omtfParams->globalPhiStartMap();
  const std::vector<L1TMuonOverlapParams::RefHitNode> *refHitMap = omtfParams->refHitMap();
  const std::vector<L1TMuonOverlapParams::LayerInputNode> *layerInputMap = omtfParams->layerInputMap();
  unsigned int tmpIndex = 0;  
  for(unsigned int iProcessor=0;iProcessor<nProcessors();++iProcessor){
    for(unsigned int iRefLayer=0;iRefLayer<nRefLayers();++iRefLayer){     
      int iPhiStart = phiStartMap->at(iRefLayer+iProcessor*nRefLayers());
      processorPhiVsRefLayer[iProcessor][iRefLayer] = iPhiStart;
    }
    for(unsigned int iRefHit=0;iRefHit<nRefHits();++iRefHit){
      int iPhiMin = refHitMap->at(iRefHit+iProcessor*nRefHits()).iPhiMin;
      int iPhiMax = refHitMap->at(iRefHit+iProcessor*nRefHits()).iPhiMax;
      unsigned int iInput = refHitMap->at(iRefHit+iProcessor*nRefHits()).iInput;
      unsigned int iRegion = refHitMap->at(iRefHit+iProcessor*nRefHits()).iRegion;
      unsigned int iRefLayer = refHitMap->at(iRefHit+iProcessor*nRefHits()).iRefLayer;
      regionPhisVsRefLayerVsInput[iInput][iRefLayer][iRegion] = std::pair<int,int>(iPhiMin,iPhiMax);
      refHitsDefs[iProcessor][iRefHit] = RefHitDef(iInput,iPhiMin,iPhiMax,iRegion,iRefLayer);
    }
    for(unsigned int iLogicRegion=0;iLogicRegion<nLogicRegions();++iLogicRegion){
      for(unsigned int iLayer=0;iLayer<nLayers();++iLayer){
	tmpIndex = iLayer+iLogicRegion*nLayers() + iProcessor*nLogicRegions()*nLayers();
	unsigned int iFirstInput = layerInputMap->at(tmpIndex).iFirstInput;
	unsigned int nInputsInRegion = layerInputMap->at(tmpIndex).nInputs;
	connections[iProcessor][iLogicRegion][iLayer] = std::pair<unsigned int, unsigned int>(iFirstInput,nInputsInRegion);
	///Symetrize connections. Use th same connections for all processors
	if(iProcessor!=0) connections[iProcessor][iLogicRegion][iLayer] = connections[0][iLogicRegion][iLayer];
      }
    }  
  }

  initCounterMatrices();
  
}
///////////////////////////////////////////////
///////////////////////////////////////////////
std::ostream & operator << (std::ostream &out, const OMTFConfiguration & aConfig){


  out<<"nLayers(): "<<aConfig.nLayers()
     <<" nHitsPerLayer(): "<<aConfig.nHitsPerLayer()
     <<" nRefLayers(): "<<aConfig.nRefLayers()
     <<" nPdfAddrBits: "<<aConfig.nPdfAddrBits()
     <<" nPdfValBits: "<<aConfig.nPdfValBits()
     <<std::endl;

  for(unsigned int iProcessor = 0;iProcessor<aConfig.nProcessors(); ++iProcessor){
    out<<"Processor: "<<iProcessor;
    for(unsigned int iRefLayer=0;iRefLayer<aConfig.nRefLayers();++iRefLayer){
      out<<" "<<aConfig.processorPhiVsRefLayer[iProcessor][iRefLayer];
    }
    out<<std::endl;
  }
  
  return out;

}
///////////////////////////////////////////////
///////////////////////////////////////////////
bool OMTFConfiguration::isInRegionRange(int iPhiStart,
				unsigned int coneSize,
				int iPhi) const {

  if(iPhi<0) iPhi+=nPhiBins();
  if(iPhiStart<0) iPhiStart+=nPhiBins();

  if(iPhiStart+(int)coneSize<(int)nPhiBins()){
    return iPhiStart<=iPhi && iPhiStart+(int)coneSize>iPhi;
  }
  else if(iPhi>(int)nPhiBins()/2){
    return iPhiStart<=iPhi;
  }
  else if(iPhi<(int)nPhiBins()/2){
    return iPhi<iPhiStart+(int)coneSize-(int)nPhiBins();
  }
  return false;
}
///////////////////////////////////////////////
///////////////////////////////////////////////
unsigned int OMTFConfiguration::getRegionNumberFromMap(unsigned int iInput,
						       unsigned int iRefLayer,						       
						       int iPhi) const {

  for(unsigned int iRegion=0;iRegion<nLogicRegions();++iRegion){
    if(iPhi>=regionPhisVsRefLayerVsInput[iInput][iRefLayer][iRegion].first &&
       iPhi<=regionPhisVsRefLayerVsInput[iInput][iRefLayer][iRegion].second)
      return iRegion;
  }

  return 99;
}
///////////////////////////////////////////////
///////////////////////////////////////////////
int OMTFConfiguration::globalPhiStart(unsigned int iProcessor) const {

  return *std::min_element(processorPhiVsRefLayer[iProcessor].begin(),
			   processorPhiVsRefLayer[iProcessor].end());

}
///////////////////////////////////////////////
///////////////////////////////////////////////
uint32_t OMTFConfiguration::getLayerNumber(uint32_t rawId) const {

  uint32_t aLayer = 0;
  
  DetId detId(rawId);
  if (detId.det() != DetId::Muon){
    std::cout << "PROBLEM: hit in unknown Det, detID: "<<detId.det()<<std::endl;
    return rawId;
  }

  switch (detId.subdetId()) {
  case MuonSubdetId::RPC: {
    RPCDetId aId(rawId);
    bool isBarrel = (aId.region()==0);
    if(isBarrel) aLayer = aId.station() <=2  ? 
		   2*( aId.station()-1)+ aId.layer() 
		   : aId.station()+2;
    else aLayer = aId.station(); 
    aLayer+= 10*(!isBarrel);
    break;
  }
  case MuonSubdetId::DT: {
    DTChamberId dt(rawId);
    aLayer = dt.station();
    break;
  }
  case MuonSubdetId::CSC: {
    CSCDetId csc(rawId);
    aLayer = csc.station();
    if(csc.ring()==2 && csc.station()==1) aLayer = 1811;//1811 = 2011 - 200, as we want to get 2011 for this chamber.
    if(csc.station()==4) aLayer = 4;
    break;
  }
  }  

  int hwNumber = aLayer+100*detId.subdetId();

  return hwNumber;
}
///////////////////////////////////////////////
///////////////////////////////////////////////
