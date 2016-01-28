#include <iostream>
#include <algorithm>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"

#include "L1Trigger/L1TMuonOverlap/interface/OMTFConfiguration.h"
#include "L1Trigger/L1TMuonOverlap/interface/XMLConfigReader.h"

float OMTFConfiguration::minPdfVal;
unsigned int OMTFConfiguration::nLayers;
unsigned int OMTFConfiguration::nHitsPerLayer;
unsigned int OMTFConfiguration::nRefLayers;
unsigned int OMTFConfiguration::nPdfAddrBits;
unsigned int OMTFConfiguration::nPdfValBits;
unsigned int OMTFConfiguration::nPhiBits;
unsigned int OMTFConfiguration::nPhiBins;
unsigned int OMTFConfiguration::nRefHits;
unsigned int OMTFConfiguration::nTestRefHits;
unsigned int OMTFConfiguration::nProcessors;
unsigned int OMTFConfiguration::nLogicRegions;
unsigned int OMTFConfiguration::nInputs;
unsigned int OMTFConfiguration::nGoldenPatterns;

std::map<int,int> OMTFConfiguration::hwToLogicLayer;
std::map<int,int> OMTFConfiguration::logicToHwLayer;
std::map<int,int> OMTFConfiguration::logicToLogic;
std::vector<int> OMTFConfiguration::refToLogicNumber;
std::set<int> OMTFConfiguration::bendingLayers;
std::vector<std::vector<int> > OMTFConfiguration::processorPhiVsRefLayer;
OMTFConfiguration::vector3D_A OMTFConfiguration::connections;
std::vector<std::vector<std::vector<std::pair<int,int> > > >OMTFConfiguration::regionPhisVsRefLayerVsProcessor;

std::vector<std::vector<RefHitDef> >OMTFConfiguration::refHitsDefs;

OMTFConfiguration::vector4D OMTFConfiguration::measurements4D;
OMTFConfiguration::vector4D OMTFConfiguration::measurements4Dref;

std::vector<unsigned int> OMTFConfiguration::barrelMin;
std::vector<unsigned int> OMTFConfiguration::barrelMax;

std::vector<unsigned int> OMTFConfiguration::endcap10DegMin;
std::vector<unsigned int> OMTFConfiguration::endcap10DegMax;

std::vector<unsigned int> OMTFConfiguration::endcap20DegMin;
std::vector<unsigned int> OMTFConfiguration::endcap20DegMax;
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
OMTFConfiguration::OMTFConfiguration(const edm::ParameterSet & theConfig){

  if(theConfig.getParameter<bool>("configFromXML")){  
    if (!theConfig.exists("configXMLFile") ) return;
    std::string fName = theConfig.getParameter<edm::FileInPath>("configXMLFile").fullPath();

    XMLConfigReader myReader;
    myReader.setConfigFile(fName);
    configure(&myReader);
  }
}
///////////////////////////////////////////////
///////////////////////////////////////////////
void OMTFConfiguration::initCounterMatrices(){
  
  ///Vector of all inputs
  std::vector<int> aLayer1D(nInputs,0);

  ///Vector of all layers 
  OMTFConfiguration::vector2D aLayer2D;
  aLayer2D.assign(nLayers,aLayer1D);

  ///Vector of all logic cones
  OMTFConfiguration::vector3D aLayer3D;
  aLayer3D.assign(nLogicRegions,aLayer2D);

  ///Vector of all processors
  measurements4D.assign(nProcessors,aLayer3D);
  measurements4Dref.assign(nProcessors,aLayer3D);
}
///////////////////////////////////////////////
///////////////////////////////////////////////
void OMTFConfiguration::configure(XMLConfigReader *aReader){

 aReader->readConfig(this);
 initCounterMatrices();

}
///////////////////////////////////////////////
///////////////////////////////////////////////
void OMTFConfiguration::configure(std::shared_ptr<L1TMuonOverlapParams> omtfParams){

  ///Set global parameters
  minPdfVal = 0.001;
  nPdfAddrBits = omtfParams->nPdfAddrBits();  
  nPdfValBits = omtfParams->nPdfValBits();
  nHitsPerLayer = omtfParams->nHitsPerLayer();
  nPhiBits = omtfParams->nPhiBits();
  nPhiBins = omtfParams->nPhiBins();
  nRefHits = omtfParams->nRefHits();
  nTestRefHits = omtfParams->nTestRefHits();
  nProcessors = omtfParams->nProcessors();
  nLogicRegions = omtfParams->nLogicRegions();
  nInputs = omtfParams->nInputs();
  nLayers = omtfParams->nLayers();
  nRefLayers = omtfParams->nRefLayers();
  nGoldenPatterns = omtfParams->nGoldenPatterns();

  ///Set chamber sectors connections to logic processros.
  barrelMin.resize(OMTFConfiguration::nProcessors);
  endcap10DegMin.resize(OMTFConfiguration::nProcessors);
  endcap20DegMin.resize(OMTFConfiguration::nProcessors);

  barrelMax.resize(OMTFConfiguration::nProcessors);
  endcap10DegMax.resize(OMTFConfiguration::nProcessors);
  endcap20DegMax.resize(OMTFConfiguration::nProcessors);

  std::vector<int> *connectedSectorsStartVec =  omtfParams->connectedSectorsStart();
  std::vector<int> *connectedSectorsEndVec =  omtfParams->connectedSectorsEnd();

  std::copy(connectedSectorsStartVec->begin(), connectedSectorsStartVec->begin()+6, barrelMin.begin());  
  std::copy(connectedSectorsStartVec->begin()+6, connectedSectorsStartVec->begin()+12, endcap10DegMin.begin());
  std::copy(connectedSectorsStartVec->begin()+12, connectedSectorsStartVec->end(), endcap20DegMin.begin());

  std::copy(connectedSectorsEndVec->begin(), connectedSectorsEndVec->begin()+6, barrelMax.begin());
  std::copy(connectedSectorsEndVec->begin()+6, connectedSectorsEndVec->begin()+12, endcap10DegMax.begin());
  std::copy(connectedSectorsEndVec->begin()+12, connectedSectorsEndVec->end(), endcap20DegMax.begin());

  ///Set connections tables
  std::vector<L1TMuonOverlapParams::LayerMapNode> *layerMap = omtfParams->layerMap();

  for(unsigned int iLayer=0;iLayer<OMTFConfiguration::nLayers;++iLayer){
    L1TMuonOverlapParams::LayerMapNode aNode = layerMap->at(iLayer);    
    hwToLogicLayer[aNode.hwNumber] = aNode.logicNumber;
    logicToHwLayer[aNode.logicNumber] = aNode.hwNumber;
    logicToLogic[aNode.logicNumber] = aNode.connectedToLayer;    
    if(aNode.bendingLayer) bendingLayers.insert(aNode.logicNumber);        
  }
  /////
  refToLogicNumber.resize(nRefLayers);
  
  std::vector<L1TMuonOverlapParams::RefLayerMapNode> *refLayerMap = omtfParams->refLayerMap();
  for(unsigned int iRefLayer=0;iRefLayer<OMTFConfiguration::nRefLayers;++iRefLayer){
    L1TMuonOverlapParams::RefLayerMapNode aNode = refLayerMap->at(iRefLayer);    
    refToLogicNumber[aNode.refLayer] = aNode.logicNumber;
  }
  /////
  std::vector<int> vector1D(OMTFConfiguration::nRefLayers,OMTFConfiguration::nPhiBins);
  processorPhiVsRefLayer.assign(OMTFConfiguration::nProcessors,vector1D);

  ///connections tables for each processor each logic cone
  ///Vector of all layers
  OMTFConfiguration::vector1D_A aLayer1D(OMTFConfiguration::nLayers);
  ///Vector of all logic cones
  OMTFConfiguration::vector2D_A aLayer2D;
  aLayer2D.assign(OMTFConfiguration::nLogicRegions,aLayer1D);
  ///Vector of all processors
  connections.assign(OMTFConfiguration::nProcessors,aLayer2D);

  ///Starting phis of each region
  ///Vector of all regions in one processor
  std::vector<std::pair<int,int> > aRefHit1D(OMTFConfiguration::nLogicRegions,std::pair<int,int>(9999,9999));
  ///Vector of all reflayers
  std::vector<std::vector<std::pair<int,int> > > aRefHit2D;
  aRefHit2D.assign(OMTFConfiguration::nRefLayers,aRefHit1D);
  ///Vector of all processors
  regionPhisVsRefLayerVsProcessor.assign(OMTFConfiguration::nProcessors,aRefHit2D);

  //Vector of ref hit definitions
  std::vector<RefHitDef> aRefHitsDefs(OMTFConfiguration::nRefHits);
  ///Vector of all processros
  refHitsDefs.assign(OMTFConfiguration::nProcessors,aRefHitsDefs);

  std::vector<int> *phiStartMap =  omtfParams->globalPhiStartMap();
  std::vector<L1TMuonOverlapParams::RefHitNode> *refHitMap = omtfParams->refHitMap();
  std::vector<L1TMuonOverlapParams::LayerInputNode> *layerInputMap = omtfParams->layerInputMap();
  unsigned int tmpIndex = 0;  
  for(unsigned int iProcessor=0;iProcessor<OMTFConfiguration::nProcessors;++iProcessor){
    for(unsigned int iRefLayer=0;iRefLayer<OMTFConfiguration::nRefLayers;++iRefLayer){     
      int iPhiStart = phiStartMap->at(iRefLayer+iProcessor*OMTFConfiguration::nRefLayers);
      processorPhiVsRefLayer[iProcessor][iRefLayer] = iPhiStart;
    }
    for(unsigned int iRefHit=0;iRefHit<OMTFConfiguration::nRefHits;++iRefHit){
      int iPhiMin = refHitMap->at(iRefHit+iProcessor*OMTFConfiguration::nRefHits).iPhiMin;
      int iPhiMax = refHitMap->at(iRefHit+iProcessor*OMTFConfiguration::nRefHits).iPhiMax;
      unsigned int iInput = refHitMap->at(iRefHit+iProcessor*OMTFConfiguration::nRefHits).iInput;
      unsigned int iRegion = refHitMap->at(iRefHit+iProcessor*OMTFConfiguration::nRefHits).iRegion;
      unsigned int iRefLayer = refHitMap->at(iRefHit+iProcessor*OMTFConfiguration::nRefHits).iRefLayer;
      regionPhisVsRefLayerVsProcessor[iProcessor][iRefLayer][iRegion] = std::pair<int,int>(iPhiMin,iPhiMax);
      refHitsDefs[iProcessor][iRefHit] = RefHitDef(iInput,iPhiMin,iPhiMax,iRegion,iRefLayer);
    }
    for(unsigned int iLogicRegion=0;iLogicRegion<OMTFConfiguration::nLogicRegions;++iLogicRegion){
      for(unsigned int iLayer=0;iLayer<OMTFConfiguration::nLayers;++iLayer){
	tmpIndex = iLayer+iLogicRegion*OMTFConfiguration::nLayers + iProcessor*OMTFConfiguration::nLogicRegions*OMTFConfiguration::nLayers;
	unsigned int iFirstInput = layerInputMap->at(tmpIndex).iFirstInput;
	unsigned int nInputs = layerInputMap->at(tmpIndex).nInputs;
	OMTFConfiguration::connections[iProcessor][iLogicRegion][iLayer] = std::pair<unsigned int, unsigned int>(iFirstInput,nInputs);
	///Symetrize connections. Use th same connections for all processors
	if(iProcessor!=0) OMTFConfiguration::connections[iProcessor][iLogicRegion][iLayer] = OMTFConfiguration::connections[0][iLogicRegion][iLayer];
      }
    }  
  }

  initCounterMatrices();
  
}
///////////////////////////////////////////////
///////////////////////////////////////////////
std::ostream & operator << (std::ostream &out, const OMTFConfiguration & aConfig){


  out<<"nLayers: "<<aConfig.nLayers
     <<" nHitsPerLayer: "<<aConfig.nHitsPerLayer
     <<" nRefLayers: "<<aConfig.nRefLayers
     <<" nPdfAddrBits: "<<aConfig.nPdfAddrBits
     <<" nPdfValBits: "<<aConfig.nPdfValBits
     <<std::endl;

  for(unsigned int iProcessor = 0;iProcessor<aConfig.nProcessors; ++iProcessor){
    out<<"Processor: "<<iProcessor;
    for(unsigned int iRefLayer=0;iRefLayer<aConfig.nRefLayers;++iRefLayer){
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
				int iPhi){

  if(iPhi<0) iPhi+=OMTFConfiguration::nPhiBins;
  if(iPhiStart<0) iPhiStart+=OMTFConfiguration::nPhiBins;

  if(iPhiStart+(int)coneSize<(int)OMTFConfiguration::nPhiBins){
    return iPhiStart<=iPhi && iPhiStart+(int)coneSize>iPhi;
  }
  else if(iPhi>(int)OMTFConfiguration::nPhiBins/2){
    return iPhiStart<=iPhi;
  }
  else if(iPhi<(int)OMTFConfiguration::nPhiBins/2){
    return iPhi<iPhiStart+(int)coneSize-(int)OMTFConfiguration::nPhiBins;
  }
  return false;
}
///////////////////////////////////////////////
///////////////////////////////////////////////
unsigned int OMTFConfiguration::getRegionNumber(unsigned int iProcessor,
					  unsigned int iRefLayer,
					  int iPhi){

  if(iPhi>=(int)OMTFConfiguration::nPhiBins) return 99;

  float logicRegionWidth = 360.0/(OMTFConfiguration::nLogicRegions*OMTFConfiguration::nProcessors);
  unsigned int logicRegionSize = logicRegionWidth/360.0*OMTFConfiguration::nPhiBins;
  
  unsigned int iRegion = 0;
  int iPhiStart = OMTFConfiguration::processorPhiVsRefLayer[iProcessor][iRefLayer];
  
  ///FIX ME 2Pi wrapping  
  while(!OMTFConfiguration::isInRegionRange(iPhiStart,logicRegionSize,iPhi) && iRegion<OMTFConfiguration::nLogicRegions){
    ++iRegion;
    iPhiStart+=logicRegionSize;    
  }
  
  if(iRegion>OMTFConfiguration::nLogicRegions-1) iRegion = 99;  
  return iRegion;
}
///////////////////////////////////////////////
///////////////////////////////////////////////
unsigned int OMTFConfiguration::getRegionNumberFromMap(unsigned int iProcessor,
						       unsigned int iRefLayer,
						       int iPhi){
  for(unsigned int iRegion=0;iRegion<OMTFConfiguration::nLogicRegions;++iRegion){
    if(iPhi>=OMTFConfiguration::regionPhisVsRefLayerVsProcessor[iProcessor][iRefLayer][iRegion].first &&
       iPhi<=OMTFConfiguration::regionPhisVsRefLayerVsProcessor[iProcessor][iRefLayer][iRegion].second)
      return iRegion;
  }

  return 99;
}
///////////////////////////////////////////////
///////////////////////////////////////////////
int OMTFConfiguration::globalPhiStart(unsigned int iProcessor){

  return *std::min_element(OMTFConfiguration::processorPhiVsRefLayer[iProcessor].begin(),
			   OMTFConfiguration::processorPhiVsRefLayer[iProcessor].end());

}
///////////////////////////////////////////////
///////////////////////////////////////////////
uint32_t OMTFConfiguration::getLayerNumber(uint32_t rawId){

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
    if(csc.ring()==2 && csc.station()==1) aLayer = 4;  /////AK TEST
    if(csc.station()==4) aLayer = 5;  /////UGLY, has to match the TEST above
    break;
  }
  }  

  int hwNumber = aLayer+100*detId.subdetId();

  return hwNumber;
}
///////////////////////////////////////////////
///////////////////////////////////////////////
