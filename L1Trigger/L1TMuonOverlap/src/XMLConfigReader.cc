#include <iostream>
#include <cmath>
#include <algorithm>
#include <utility>

#include "L1Trigger/L1TMuonOverlap/interface/XMLConfigReader.h"
#include "L1Trigger/L1TMuonOverlap/interface/GoldenPattern.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFinput.h"

#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "xercesc/framework/StdOutFormatTarget.hpp"
#include "xercesc/framework/LocalFileFormatTarget.hpp"
#include "xercesc/parsers/XercesDOMParser.hpp"
#include "xercesc/dom/DOM.hpp"
#include "xercesc/dom/DOMException.hpp"
#include "xercesc/dom/DOMImplementation.hpp"
#include "xercesc/sax/HandlerBase.hpp"
#include "xercesc/util/XMLString.hpp"
#include "xercesc/util/PlatformUtils.hpp"
#include "xercesc/util/XercesDefs.hpp"
XERCES_CPP_NAMESPACE_USE

#include "L1Trigger/RPCTrigger/interface/RPCConst.h"
//////////////////////////////////
// XMLConfigReader
//////////////////////////////////
inline std::string _toString(XMLCh const* toTranscode) {
std::string tmp(xercesc::XMLString::transcode(toTranscode));
return tmp;
}

inline XMLCh*  _toDOMS(std::string temp) {
  XMLCh* buff = XMLString::transcode(temp.c_str());
  return  buff;
}
////////////////////////////////////
////////////////////////////////////
XMLConfigReader::XMLConfigReader(){

  XMLPlatformUtils::Initialize();
  
  ///Initialise XML parser  
  parser = new XercesDOMParser(); 
  parser->setValidationScheme(XercesDOMParser::Val_Auto);
  parser->setDoNamespaces(false);

  doc = 0;  
}
//////////////////////////////////////////////////
//////////////////////////////////////////////////
void XMLConfigReader::readLUT(l1t::LUT *lut,const L1TMuonOverlapParams & aConfig, const std::string & type){

  std::stringstream strStream;
  int totalInWidth = 6;
  int outWidth = 6;

  if(type=="iCharge") outWidth = 1;
  if(type=="iEta") outWidth = 2;
  if(type=="iPt") outWidth = 9;
  if(type=="meanDistPhi"){
    outWidth = 11;
    totalInWidth = 14;
  }
  if(type=="pdf"){
    outWidth = 6;
    totalInWidth = 20;
  }
  
  ///Prepare the header 
  strStream <<"#<header> V1 "<<totalInWidth<<" "<<outWidth<<" </header> "<<std::endl;
  
  ///Fill payload string  
  const std::vector<GoldenPattern *> & aGPs = readPatterns(aConfig);
  unsigned int in = 0;
  int out = 0;
  for(auto it: aGPs){
    if(type=="iCharge") out = it->key().theCharge==-1 ? 0:1;
    if(type=="iEta") out = it->key().theEtaCode;
    if(type=="iPt") out = it->key().thePtCode;
    if(type=="meanDistPhi"){
      for(unsigned int iLayer = 0;iLayer<(unsigned) aConfig.nLayers();++iLayer){
	for(unsigned int iRefLayer=0;iRefLayer<(unsigned) aConfig.nRefLayers();++iRefLayer){
	  out = (1<<(outWidth-1)) + it->meanDistPhiValue(iLayer,iRefLayer);
	  strStream<<in<<" "<<out<<std::endl;
	  ++in;
	}
      }
    }
    if(type=="pdf"){
      for(unsigned int iLayer = 0;iLayer<(unsigned)aConfig.nLayers();++iLayer){
	for(unsigned int iRefLayer=0;iRefLayer<(unsigned)aConfig.nRefLayers();++iRefLayer){
	  for(unsigned int iPdf=0;iPdf<exp2(aConfig.nPdfAddrBits());++iPdf){
	    out = it->pdfValue(iLayer,iRefLayer,iPdf);
	    strStream<<in<<" "<<out<<std::endl;
	    ++in;
	  }
	}
      }
    }
    if(type!="meanDistPhi" && type!="pdf"){
      strStream<<in<<" "<<out<<std::endl;
      ++in;
    }
  } 
  ///Read the data into LUT
  lut->read(strStream);
}
//////////////////////////////////////////////////
//////////////////////////////////////////////////
std::vector<GoldenPattern*> XMLConfigReader::readPatterns(const L1TMuonOverlapParams & aConfig){

  aGPs.clear();
  
  parser->parse(patternsFile.c_str()); 
  xercesc::DOMDocument* doc = parser->getDocument();
  assert(doc);

  unsigned int nElem = doc->getElementsByTagName(_toDOMS("GP"))->getLength();
  if(nElem<1){
    edm::LogError("critical")<<"Problem parsing XML file "<<patternsFile<<std::endl;
    edm::LogError("critical")<<"No GoldenPattern items: GP found"<<std::endl;
    return aGPs;
  }

  DOMNode *aNode = 0;
  DOMElement* aGPElement = 0;
  for(unsigned int iItem=0;iItem<nElem;++iItem){
    aNode = doc->getElementsByTagName(_toDOMS("GP"))->item(iItem);
    aGPElement = static_cast<DOMElement *>(aNode);

    std::ostringstream stringStr;
    GoldenPattern *aGP;
    for(unsigned int index = 1;index<5;++index){
      stringStr.str("");
      stringStr<<"iPt"<<index;
      ///Patterns XML format backward compatibility. Can use both packed by 4, or by 1 XML files.      
      if(aGPElement->getAttributeNode(_toDOMS(stringStr.str().c_str()))){
	aGP = buildGP(aGPElement, aConfig, index);
	if(aGP) aGPs.push_back(aGP);
      }
      else{
	aGP = buildGP(aGPElement, aConfig);
	if(aGP) aGPs.push_back(aGP);
	break;
      }
    }
  }
  delete doc;

  return aGPs;
}
//////////////////////////////////////////////////
//////////////////////////////////////////////////
GoldenPattern * XMLConfigReader::buildGP(DOMElement* aGPElement,
					 const L1TMuonOverlapParams & aConfig,
					 unsigned int index){

  std::ostringstream stringStr; 
  if(index>0) stringStr<<"iPt"<<index;
  else stringStr.str("iPt");
  
  unsigned int iPt = std::atoi(_toString(aGPElement->getAttribute(_toDOMS(stringStr.str().c_str()))).c_str());
  if(iPt==0) return 0;
  
  int iEta = std::atoi(_toString(aGPElement->getAttribute(_toDOMS("iEta"))).c_str());
  int iCharge = std::atoi(_toString(aGPElement->getAttribute(_toDOMS("iCharge"))).c_str());
  int val = 0;
  unsigned int nLayers = aGPElement->getElementsByTagName(_toDOMS("Layer"))->getLength();
  assert(nLayers==(unsigned) aConfig.nLayers());

  DOMNode *aNode = 0;
  DOMElement* aLayerElement = 0;
  DOMElement* aItemElement = 0;
  GoldenPattern::vector2D meanDistPhi2D(nLayers);
  GoldenPattern::vector1D pdf1D(exp2(aConfig.nPdfAddrBits()));
  GoldenPattern::vector3D pdf3D(aConfig.nLayers());
  GoldenPattern::vector2D pdf2D(aConfig.nRefLayers());
  ///Loop over layers
  for(unsigned int iLayer=0;iLayer<nLayers;++iLayer){
    aNode = aGPElement->getElementsByTagName(_toDOMS("Layer"))->item(iLayer);
    aLayerElement = static_cast<DOMElement *>(aNode); 
    ///MeanDistPhi vector
    unsigned int nItems = aLayerElement->getElementsByTagName(_toDOMS("RefLayer"))->getLength();
    assert(nItems==(unsigned) aConfig.nRefLayers());
    GoldenPattern::vector1D meanDistPhi1D(nItems);
    for(unsigned int iItem=0;iItem<nItems;++iItem){
      aNode = aLayerElement->getElementsByTagName(_toDOMS("RefLayer"))->item(iItem);
      aItemElement = static_cast<DOMElement *>(aNode); 
      val = std::atoi(_toString(aItemElement->getAttribute(_toDOMS("meanDistPhi"))).c_str());
      meanDistPhi1D[iItem] = val;
    }
    meanDistPhi2D[iLayer] = meanDistPhi1D;

    ///PDF vector
    stringStr.str("");
    if(index>0) stringStr<<"value"<<index;
    else stringStr.str("value");    
    nItems = aLayerElement->getElementsByTagName(_toDOMS("PDF"))->getLength();
    assert(nItems==aConfig.nRefLayers()*exp2(aConfig.nPdfAddrBits()));
    for(unsigned int iRefLayer=0;iRefLayer<(unsigned) aConfig.nRefLayers();++iRefLayer){
      pdf1D.assign(exp2(aConfig.nPdfAddrBits()),0);
      for(unsigned int iPdf=0;iPdf<exp2(aConfig.nPdfAddrBits());++iPdf){
	aNode = aLayerElement->getElementsByTagName(_toDOMS("PDF"))->item(iRefLayer*exp2(aConfig.nPdfAddrBits())+iPdf);
	aItemElement = static_cast<DOMElement *>(aNode);
	val = std::atoi(_toString(aItemElement->getAttribute(_toDOMS(stringStr.str().c_str()))).c_str());
	pdf1D[iPdf] = val;
      }
      pdf2D[iRefLayer] = pdf1D;
    }
    pdf3D[iLayer] = pdf2D;
  }

  Key aKey(iEta,iPt,iCharge);
  GoldenPattern *aGP = new GoldenPattern(aKey,0);
  aGP->setMeanDistPhi(meanDistPhi2D);
  aGP->setPdf(pdf3D);

  return aGP;
}
//////////////////////////////////////////////////
//////////////////////////////////////////////////
std::vector<std::vector<int> > XMLConfigReader::readEvent(unsigned int iEvent,
							  unsigned int iProcessor,
							  bool readEta){

  return OMTFinput::vector2D();
  
  /*
  if(!doc){
    parser->parse(eventsFile.c_str()); 
    doc = parser->getDocument();
  }
  assert(doc);

  OMTFinput::vector1D input1D(14,m_omtf_config->nPhiBins);
  OMTFinput::vector2D input2D(m_omtf_config->nLayers);
  unsigned int nElem = doc->getElementsByTagName(_toDOMS("OMTF_Events"))->getLength();
  assert(nElem==1);
 
  DOMNode *aNode = doc->getElementsByTagName(_toDOMS("OMTF_Events"))->item(0);
  DOMElement* aOMTFElement = static_cast<DOMElement *>(aNode); 
  DOMElement* aEventElement = 0;
  DOMElement* aBxElement = 0;
  DOMElement* aProcElement = 0;
  DOMElement* aLayerElement = 0;
  DOMElement* aHitElement = 0;
  unsigned int aLogicLayer = m_omtf_config->nLayers+1;
  int val = 0, input=0;

  nElem = aOMTFElement->getElementsByTagName(_toDOMS("Event"))->getLength();
   if(nElem<iEvent){
    edm::LogError("critical")<<"Problem parsing XML file "<<eventsFile<<std::endl;
    edm::LogError("critical")<<"not enough events found: "<<nElem<<std::endl;
    assert(nElem>=iEvent);
  }
 
  aNode = aOMTFElement->getElementsByTagName(_toDOMS("Event"))->item(iEvent);
  aEventElement = static_cast<DOMElement *>(aNode); 
  
  unsigned int nBX = aEventElement->getElementsByTagName(_toDOMS("bx"))->getLength();
  assert(nBX>0);
  aNode = aEventElement->getElementsByTagName(_toDOMS("bx"))->item(0);
  aBxElement = static_cast<DOMElement *>(aNode); 

  unsigned int nProc = aEventElement->getElementsByTagName(_toDOMS("Processor"))->getLength();
  unsigned int aProcID = 99;
  assert(nProc>=iProcessor);
  for(unsigned int aProc=0;aProc<nProc;++aProc){
    aNode = aBxElement->getElementsByTagName(_toDOMS("Processor"))->item(aProc);
    aProcElement = static_cast<DOMElement *>(aNode); 
    aProcID = std::atoi(_toString(aProcElement->getAttribute(_toDOMS("iProcessor"))).c_str());
    if(aProcID==iProcessor) break;
  }
  if(aProcID!=iProcessor) return input2D;
     
  unsigned int nLayersHit = aProcElement->getElementsByTagName(_toDOMS("Layer"))->getLength();    
  assert(nLayersHit<=m_omtf_config->nLayers);
  
  input2D.assign(m_omtf_config->nLayers,input1D);  
  for(unsigned int iLayer=0;iLayer<nLayersHit;++iLayer){
    aNode = aProcElement->getElementsByTagName(_toDOMS("Layer"))->item(iLayer);
    aLayerElement = static_cast<DOMElement *>(aNode); 
    aLogicLayer = std::atoi(_toString(aLayerElement->getAttribute(_toDOMS("iLayer"))).c_str());
    nElem = aLayerElement->getElementsByTagName(_toDOMS("Hit"))->getLength();     
    input1D.assign(14,m_omtf_config->nPhiBins);

    for(unsigned int iHit=0;iHit<nElem;++iHit){
      aNode = aLayerElement->getElementsByTagName(_toDOMS("Hit"))->item(iHit);
      aHitElement = static_cast<DOMElement *>(aNode); 
      val = std::atoi(_toString(aHitElement->getAttribute(_toDOMS("iPhi"))).c_str());
      if(readEta) val = std::atoi(_toString(aHitElement->getAttribute(_toDOMS("iEta"))).c_str());
      input = std::atoi(_toString(aHitElement->getAttribute(_toDOMS("iInput"))).c_str());
      input1D[input] = val;
    }
    input2D[aLogicLayer] = input1D;
  }

  //delete doc;
  return input2D;
  */
}
//////////////////////////////////////////////////
//////////////////////////////////////////////////
void XMLConfigReader::readConfig(L1TMuonOverlapParams *aConfig) const{

 parser->parse(configFile.c_str()); 
  xercesc::DOMDocument* doc = parser->getDocument();
  assert(doc);
  unsigned int nElem = doc->getElementsByTagName(_toDOMS("OMTF"))->getLength();
  if(nElem!=1){
    edm::LogError("critical")<<"Problem parsing XML file "<<configFile<<std::endl;
    assert(nElem==1);
  }
  DOMNode *aNode = doc->getElementsByTagName(_toDOMS("OMTF"))->item(0);
  DOMElement* aOMTFElement = static_cast<DOMElement *>(aNode);

  unsigned int version = std::stoul(_toString(aOMTFElement->getAttribute(_toDOMS("version"))), nullptr, 16);
  aConfig->setFwVersion(version);

  ///Addresing bits numbers
  nElem = aOMTFElement->getElementsByTagName(_toDOMS("GlobalData"))->getLength();
  assert(nElem==1);
  aNode = aOMTFElement->getElementsByTagName(_toDOMS("GlobalData"))->item(0);
  DOMElement* aElement = static_cast<DOMElement *>(aNode); 

  unsigned int nPdfAddrBits = std::atoi(_toString(aElement->getAttribute(_toDOMS("nPdfAddrBits"))).c_str()); 
  unsigned int nPdfValBits =  std::atoi(_toString(aElement->getAttribute(_toDOMS("nPdfValBits"))).c_str()); 
  unsigned int nHitsPerLayer =  std::atoi(_toString(aElement->getAttribute(_toDOMS("nHitsPerLayer"))).c_str()); 
  unsigned int nPhiBits =  std::atoi(_toString(aElement->getAttribute(_toDOMS("nPhiBits"))).c_str()); 
  unsigned int nPhiBins =  std::atoi(_toString(aElement->getAttribute(_toDOMS("nPhiBins"))).c_str()); 
  unsigned int nRefHits =  std::atoi(_toString(aElement->getAttribute(_toDOMS("nRefHits"))).c_str()); 
  unsigned int nTestRefHits =  std::atoi(_toString(aElement->getAttribute(_toDOMS("nTestRefHits"))).c_str());
  unsigned int nProcessors =  std::atoi(_toString(aElement->getAttribute(_toDOMS("nProcessors"))).c_str());
  unsigned int nLogicRegions =  std::atoi(_toString(aElement->getAttribute(_toDOMS("nLogicRegions"))).c_str());
  unsigned int nInputs =  std::atoi(_toString(aElement->getAttribute(_toDOMS("nInputs"))).c_str());
  unsigned int nLayers =  std::atoi(_toString(aElement->getAttribute(_toDOMS("nLayers"))).c_str());
  unsigned int nRefLayers =  std::atoi(_toString(aElement->getAttribute(_toDOMS("nRefLayers"))).c_str());
  unsigned int nGoldenPatterns =  std::atoi(_toString(aElement->getAttribute(_toDOMS("nGoldenPatterns"))).c_str());

  std::vector<int> paramsVec(L1TMuonOverlapParams::GENERAL_NCONFIG);
  paramsVec[L1TMuonOverlapParams::GENERAL_ADDRBITS] = nPdfAddrBits;
  paramsVec[L1TMuonOverlapParams::GENERAL_VALBITS] = nPdfValBits;
  paramsVec[L1TMuonOverlapParams::GENERAL_HITSPERLAYER] = nHitsPerLayer;
  paramsVec[L1TMuonOverlapParams::GENERAL_PHIBITS] = nPhiBits;
  paramsVec[L1TMuonOverlapParams::GENERAL_PHIBINS] = nPhiBins;
  paramsVec[L1TMuonOverlapParams::GENERAL_NREFHITS] = nRefHits;
  paramsVec[L1TMuonOverlapParams::GENERAL_NTESTREFHITS] = nTestRefHits;
  paramsVec[L1TMuonOverlapParams::GENERAL_NPROCESSORS] = nProcessors;
  paramsVec[L1TMuonOverlapParams::GENERAL_NLOGIC_REGIONS] = nLogicRegions;
  paramsVec[L1TMuonOverlapParams::GENERAL_NINPUTS] = nInputs;
  paramsVec[L1TMuonOverlapParams::GENERAL_NLAYERS] = nLayers;
  paramsVec[L1TMuonOverlapParams::GENERAL_NREFLAYERS] = nRefLayers;
  paramsVec[L1TMuonOverlapParams::GENERAL_NGOLDENPATTERNS] = nGoldenPatterns;
  aConfig->setGeneralParams(paramsVec);

  ///Chamber sectors connections to logic processros.
  ///Start/End values for all processors, and chamber types are put into a single vector
  std::vector<int> sectorsStart(3*nProcessors), sectorsEnd(3*nProcessors);
  nElem = aOMTFElement->getElementsByTagName(_toDOMS("ConnectionMap"))->getLength();
  DOMElement* aConnectionElement = 0;
  for(uint i=0;i<nElem;++i){
    aNode = aOMTFElement->getElementsByTagName(_toDOMS("ConnectionMap"))->item(i);
    aConnectionElement = static_cast<DOMElement *>(aNode);
    unsigned int iProcessor = std::atoi(_toString(aConnectionElement->getAttribute(_toDOMS("iProcessor"))).c_str());
    unsigned int barrelMin = std::atoi(_toString(aConnectionElement->getAttribute(_toDOMS("barrelMin"))).c_str());
    unsigned int barrelMax = std::atoi(_toString(aConnectionElement->getAttribute(_toDOMS("barrelMax"))).c_str());
    unsigned int endcap10DegMin = std::atoi(_toString(aConnectionElement->getAttribute(_toDOMS("endcap10DegMin"))).c_str());
    unsigned int endcap10DegMax = std::atoi(_toString(aConnectionElement->getAttribute(_toDOMS("endcap10DegMax"))).c_str());
    unsigned int endcap20DegMin = std::atoi(_toString(aConnectionElement->getAttribute(_toDOMS("endcap20DegMin"))).c_str());
    unsigned int endcap20DegMax = std::atoi(_toString(aConnectionElement->getAttribute(_toDOMS("endcap20DegMax"))).c_str());

    sectorsStart[iProcessor] = barrelMin;
    sectorsStart[iProcessor + nProcessors] = endcap10DegMin;
    sectorsStart[iProcessor  + 2*nProcessors] = endcap20DegMin;

    sectorsEnd[iProcessor] = barrelMax;
    sectorsEnd[iProcessor + nProcessors] = endcap10DegMax;
    sectorsEnd[iProcessor + 2*nProcessors] = endcap20DegMax;       
  }  
  aConfig->setConnectedSectorsStart(sectorsStart);
  aConfig->setConnectedSectorsEnd(sectorsEnd);

    
  ///hw <-> logic numbering map
  std::vector<L1TMuonOverlapParams::LayerMapNode> aLayerMapVec;
  L1TMuonOverlapParams::LayerMapNode aLayerMapNode;
 
  nElem = aOMTFElement->getElementsByTagName(_toDOMS("LayerMap"))->getLength();
  DOMElement* aLayerElement = 0;
  for(uint i=0;i<nElem;++i){
    aNode = aOMTFElement->getElementsByTagName(_toDOMS("LayerMap"))->item(i);
    aLayerElement = static_cast<DOMElement *>(aNode); 
    unsigned int hwNumber = std::atoi(_toString(aLayerElement->getAttribute(_toDOMS("hwNumber"))).c_str());
    unsigned int logicNumber = std::atoi(_toString(aLayerElement->getAttribute(_toDOMS("logicNumber"))).c_str());
    unsigned int isBendingLayer = std::atoi(_toString(aLayerElement->getAttribute(_toDOMS("bendingLayer"))).c_str());
    unsigned int iConnectedLayer = std::atoi(_toString(aLayerElement->getAttribute(_toDOMS("connectedToLayer"))).c_str());
    aLayerMapNode.logicNumber = logicNumber;
    aLayerMapNode.hwNumber = hwNumber;
    aLayerMapNode.connectedToLayer = iConnectedLayer;
    aLayerMapNode.bendingLayer = isBendingLayer;
    aLayerMapVec.push_back(aLayerMapNode);
  }
  aConfig->setLayerMap(aLayerMapVec);

  ///ref<->logic numberig map
  std::vector<L1TMuonOverlapParams::RefLayerMapNode> aRefLayerMapVec;
  L1TMuonOverlapParams::RefLayerMapNode aRefLayerNode;
  
  nElem = aOMTFElement->getElementsByTagName(_toDOMS("RefLayerMap"))->getLength();
  DOMElement* aRefLayerElement = 0;
  for(uint i=0;i<nElem;++i){
    aNode = aOMTFElement->getElementsByTagName(_toDOMS("RefLayerMap"))->item(i);
    aRefLayerElement = static_cast<DOMElement *>(aNode); 
    unsigned int refLayer = std::atoi(_toString(aRefLayerElement->getAttribute(_toDOMS("refLayer"))).c_str());
    unsigned int logicNumber = std::atoi(_toString(aRefLayerElement->getAttribute(_toDOMS("logicNumber"))).c_str());
    aRefLayerNode.refLayer = refLayer;
    aRefLayerNode.logicNumber = logicNumber;
    aRefLayerMapVec.push_back(aRefLayerNode);
  }
  aConfig->setRefLayerMap(aRefLayerMapVec);

  std::vector<int> aGlobalPhiStartVec(nProcessors*nRefLayers);
  
  std::vector<L1TMuonOverlapParams::RefHitNode> aRefHitMapVec(nProcessors*nRefHits);
  L1TMuonOverlapParams::RefHitNode aRefHitNode;
  
  std::vector<L1TMuonOverlapParams::LayerInputNode> aLayerInputMapVec(nProcessors*nLogicRegions*nLayers);  
  L1TMuonOverlapParams::LayerInputNode aLayerInputNode;
  
  nElem = aOMTFElement->getElementsByTagName(_toDOMS("Processor"))->getLength();
  assert(nElem==nProcessors);
  DOMElement* aProcessorElement = 0;
  for(uint i=0;i<nElem;++i){
    aNode = aOMTFElement->getElementsByTagName(_toDOMS("Processor"))->item(i);
    aProcessorElement = static_cast<DOMElement *>(aNode); 
    unsigned int iProcessor = std::atoi(_toString(aProcessorElement->getAttribute(_toDOMS("iProcessor"))).c_str());
    unsigned int nElem1 = aProcessorElement->getElementsByTagName(_toDOMS("RefLayer"))->getLength();
    assert(nElem1==nRefLayers);
    DOMElement* aRefLayerElement = 0;
    for(uint ii=0;ii<nElem1;++ii){
      aNode = aProcessorElement->getElementsByTagName(_toDOMS("RefLayer"))->item(ii);
      aRefLayerElement = static_cast<DOMElement *>(aNode); 
      unsigned int iRefLayer = std::atoi(_toString(aRefLayerElement->getAttribute(_toDOMS("iRefLayer"))).c_str());
      int iPhi = std::atoi(_toString(aRefLayerElement->getAttribute(_toDOMS("iGlobalPhiStart"))).c_str());
      aGlobalPhiStartVec[iRefLayer + iProcessor*nRefLayers] = iPhi;
    }
    ///////////
    nElem1 = aProcessorElement->getElementsByTagName(_toDOMS("RefHit"))->getLength();
    assert( (iProcessor==0 && nElem1==nRefHits) || (iProcessor!=0 && nElem1==0) );
    DOMElement* aRefHitElement = 0;
    for(uint ii=0;ii<nElem1;++ii){
      aNode = aProcessorElement->getElementsByTagName(_toDOMS("RefHit"))->item(ii);
      aRefHitElement = static_cast<DOMElement *>(aNode); 
      unsigned int iRefHit = std::atoi(_toString(aRefHitElement->getAttribute(_toDOMS("iRefHit"))).c_str());
      int iPhiMin = std::atoi(_toString(aRefHitElement->getAttribute(_toDOMS("iPhiMin"))).c_str());
      int iPhiMax = std::atoi(_toString(aRefHitElement->getAttribute(_toDOMS("iPhiMax"))).c_str());
      unsigned int iInput = std::atoi(_toString(aRefHitElement->getAttribute(_toDOMS("iInput"))).c_str());
      unsigned int iRegion = std::atoi(_toString(aRefHitElement->getAttribute(_toDOMS("iRegion"))).c_str());
      unsigned int iRefLayer = std::atoi(_toString(aRefHitElement->getAttribute(_toDOMS("iRefLayer"))).c_str());

      aRefHitNode.iRefHit = iRefHit;
      aRefHitNode.iPhiMin = iPhiMin;
      aRefHitNode.iPhiMax = iPhiMax;
      aRefHitNode.iInput = iInput;
      aRefHitNode.iRegion = iRegion;
      aRefHitNode.iRefLayer = iRefLayer;
      for (unsigned int iProcessor=0; iProcessor<nProcessors; iProcessor++) aRefHitMapVec[iRefHit + iProcessor*nRefHits] = aRefHitNode;
    }
    ///////////
    unsigned int nElem2 = aProcessorElement->getElementsByTagName(_toDOMS("LogicRegion"))->getLength();
    assert( (iProcessor==0 && nElem2==nLogicRegions) || (iProcessor!=0 && nElem2==0) );
    DOMElement* aRegionElement = 0;
    for(uint ii=0;ii<nElem2;++ii){
      aNode = aProcessorElement->getElementsByTagName(_toDOMS("LogicRegion"))->item(ii);
      aRegionElement = static_cast<DOMElement *>(aNode); 
      unsigned int iRegion = std::atoi(_toString(aRegionElement->getAttribute(_toDOMS("iRegion"))).c_str());
      unsigned int nElem3 = aRegionElement->getElementsByTagName(_toDOMS("Layer"))->getLength();
      assert(nElem3==nLayers);
      DOMElement* aLayerElement = 0;
      for(uint iii=0;iii<nElem3;++iii){
  	  aNode = aRegionElement->getElementsByTagName(_toDOMS("Layer"))->item(iii);
	  aLayerElement = static_cast<DOMElement *>(aNode); 
	  unsigned int iLayer = std::atoi(_toString(aLayerElement->getAttribute(_toDOMS("iLayer"))).c_str());
	  unsigned int iFirstInput = std::atoi(_toString(aLayerElement->getAttribute(_toDOMS("iFirstInput"))).c_str());
	  unsigned int nInputs = std::atoi(_toString(aLayerElement->getAttribute(_toDOMS("nInputs"))).c_str());
	  aLayerInputNode.iLayer = iLayer;
	  aLayerInputNode.iFirstInput = iFirstInput;
	  aLayerInputNode.nInputs = nInputs;
	  for (unsigned int iProcessor=0; iProcessor<nProcessors; ++iProcessor) aLayerInputMapVec[iLayer + iRegion*nLayers + iProcessor*nLayers*nLogicRegions] = aLayerInputNode;
      }
    }   
  }

  aConfig->setGlobalPhiStartMap(aGlobalPhiStartVec);
  aConfig->setLayerInputMap(aLayerInputMapVec);
  aConfig->setRefHitMap(aRefHitMapVec);

  delete doc;
}
//////////////////////////////////////////////////
//////////////////////////////////////////////////

