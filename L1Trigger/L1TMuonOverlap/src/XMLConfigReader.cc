#include <iostream>
#include <cmath>
#include <algorithm>
#include <utility>

#include "L1Trigger/L1TMuonOverlap/interface/XMLConfigReader.h"
#include "L1Trigger/L1TMuonOverlap/interface/GoldenPattern.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFinput.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFConfiguration.h"

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
void XMLConfigReader::readLUT(l1t::LUT *lut, const std::string & type){

  std::stringstream strStream;
  int totalInWidth = 6;
  int outWidth = 6;

  if(type=="iCharge") outWidth = 1;
  if(type=="iEta") outWidth = 2;
  if(type=="iPt") outWidth = 6;
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
  const std::vector<GoldenPattern *> & aGPs = readPatterns();
  unsigned int in = 0;
  int out = 0;
  for(auto it: aGPs){
    if(type=="iCharge") out = it->key().theCharge + 1*(it->key().theCharge<0);
    if(type=="iEta") out = it->key().theEtaCode;
    if(type=="iPt") out = it->key().thePtCode;
    if(type=="meanDistPhi"){
      for(unsigned int iLayer = 0;iLayer<OMTFConfiguration::nLayers;++iLayer){
	for(unsigned int iRefLayer=0;iRefLayer<OMTFConfiguration::nRefLayers;++iRefLayer){
	  out = (1<<(outWidth-1)) + it->meanDistPhiValue(iLayer,iRefLayer);
	  strStream<<in<<" "<<out<<std::endl;
	  ++in;
	}
      }
    }
    if(type=="pdf"){
      for(unsigned int iLayer = 0;iLayer<OMTFConfiguration::nLayers;++iLayer){
	for(unsigned int iRefLayer=0;iRefLayer<OMTFConfiguration::nRefLayers;++iRefLayer){
	  for(unsigned int iPdf=0;iPdf<exp2(OMTFConfiguration::nPdfAddrBits);++iPdf){
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
std::vector<GoldenPattern*> XMLConfigReader::readPatterns(){

  if(aGPs.size()) return aGPs;
  
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
      ///Patterns XMl format backward compatibility. Can use both packed by 4, or by 1 XML files.      
      if(aGPElement->getAttributeNode(_toDOMS(stringStr.str().c_str()))){
	aGP = buildGP(aGPElement,index);
	if(aGP) aGPs.push_back(aGP);
      }
      else{
	aGPs.push_back(buildGP(aGPElement));
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
  assert(nLayers==OMTFConfiguration::nLayers);
  DOMNode *aNode = 0;
  DOMElement* aLayerElement = 0;
  DOMElement* aItemElement = 0;
  GoldenPattern::vector2D meanDistPhi2D(nLayers);
  GoldenPattern::vector1D pdf1D(exp2(OMTFConfiguration::nPdfAddrBits));
  GoldenPattern::vector3D pdf3D(OMTFConfiguration::nLayers);
  GoldenPattern::vector2D pdf2D(OMTFConfiguration::nRefLayers);
  ///Loop over layers
  for(unsigned int iLayer=0;iLayer<nLayers;++iLayer){
    aNode = aGPElement->getElementsByTagName(_toDOMS("Layer"))->item(iLayer);
    aLayerElement = static_cast<DOMElement *>(aNode); 
    ///MeanDistPhi vector
    unsigned int nItems = aLayerElement->getElementsByTagName(_toDOMS("RefLayer"))->getLength();
    assert(nItems==OMTFConfiguration::nRefLayers);
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
    assert(nItems==OMTFConfiguration::nRefLayers*exp2(OMTFConfiguration::nPdfAddrBits));
    for(unsigned int iRefLayer=0;iRefLayer<OMTFConfiguration::nRefLayers;++iRefLayer){
      pdf1D.assign(exp2(OMTFConfiguration::nPdfAddrBits),0);
      for(unsigned int iPdf=0;iPdf<exp2(OMTFConfiguration::nPdfAddrBits);++iPdf){
	aNode = aLayerElement->getElementsByTagName(_toDOMS("PDF"))->item(iRefLayer*exp2(OMTFConfiguration::nPdfAddrBits)+iPdf);
	aItemElement = static_cast<DOMElement *>(aNode);
	val = std::atoi(_toString(aItemElement->getAttribute(_toDOMS(stringStr.str().c_str()))).c_str());
	pdf1D[iPdf] = val;
      }
      pdf2D[iRefLayer] = pdf1D;
    }
    pdf3D[iLayer] = pdf2D;
  }

  Key aKey(iEta,iPt,iCharge);
  GoldenPattern *aGP = new GoldenPattern(aKey);
  aGP->setMeanDistPhi(meanDistPhi2D);
  aGP->setPdf(pdf3D);

  return aGP;
}
//////////////////////////////////////////////////
//////////////////////////////////////////////////
std::vector<std::vector<int> > XMLConfigReader::readEvent(unsigned int iEvent,
							  unsigned int iProcessor,
							  bool readEta){
  if(!doc){
    parser->parse(eventsFile.c_str()); 
    doc = parser->getDocument();
  }
  assert(doc);


  OMTFinput::vector1D input1D(14,OMTFConfiguration::nPhiBins);
  OMTFinput::vector2D input2D(OMTFConfiguration::nLayers);

  unsigned int nElem = doc->getElementsByTagName(_toDOMS("OMTF_Events"))->getLength();
  assert(nElem==1);
 
  DOMNode *aNode = doc->getElementsByTagName(_toDOMS("OMTF_Events"))->item(0);
  DOMElement* aOMTFElement = static_cast<DOMElement *>(aNode); 
  DOMElement* aEventElement = 0;
  DOMElement* aBxElement = 0;
  DOMElement* aProcElement = 0;
  DOMElement* aLayerElement = 0;
  DOMElement* aHitElement = 0;
  unsigned int aLogicLayer = OMTFConfiguration::nLayers+1;
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
  assert(nLayersHit<=OMTFConfiguration::nLayers);
  
  input2D.assign(OMTFConfiguration::nLayers,input1D);
  
  for(unsigned int iLayer=0;iLayer<nLayersHit;++iLayer){
    aNode = aProcElement->getElementsByTagName(_toDOMS("Layer"))->item(iLayer);
    aLayerElement = static_cast<DOMElement *>(aNode); 
    aLogicLayer = std::atoi(_toString(aLayerElement->getAttribute(_toDOMS("iLayer"))).c_str());
    nElem = aLayerElement->getElementsByTagName(_toDOMS("Hit"))->getLength();     
    input1D.assign(14,OMTFConfiguration::nPhiBins);
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
}
//////////////////////////////////////////////////
//////////////////////////////////////////////////
void XMLConfigReader::readConfig( L1TMuonOverlapParams *aConfig){

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

  unsigned int version = std::atoi(_toString(aOMTFElement->getAttribute(_toDOMS("version"))).c_str());
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
  std::vector<int> sectorsStart(3*6), sectorsEnd(3*6);
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
    sectorsStart[iProcessor + 6] = endcap10DegMin;
    sectorsStart[iProcessor  +12] = endcap20DegMin;

    sectorsEnd[iProcessor] = barrelMax;
    sectorsEnd[iProcessor + 6] = endcap10DegMax;
    sectorsEnd[iProcessor + 12] = endcap20DegMax;       
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
    assert(nElem1==nRefHits);
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
      aRefHitMapVec[iRefHit + iProcessor*nRefHits] = aRefHitNode;
    }
    ///////////
    unsigned int nElem2 = aProcessorElement->getElementsByTagName(_toDOMS("LogicRegion"))->getLength();
    assert(nElem2==nProcessors);
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
	aLayerInputMapVec[iLayer + iRegion*nLayers + iProcessor*nLayers*nLogicRegions] = aLayerInputNode;
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
void XMLConfigReader::readConfig(OMTFConfiguration *aConfig){
  
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

  ///Addresing bits numbers
  nElem = aOMTFElement->getElementsByTagName(_toDOMS("GlobalData"))->getLength();
  assert(nElem==1);
  aNode = aOMTFElement->getElementsByTagName(_toDOMS("GlobalData"))->item(0);
  DOMElement* aElement = static_cast<DOMElement *>(aNode); 

  float minPdfVal = 0.001;
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
  unsigned int nGoldenPatterns =  std::atoi(_toString(aElement->getAttribute(_toDOMS("nGoldenPatterns"))).c_str()); 
  OMTFConfiguration::minPdfVal = minPdfVal;
  OMTFConfiguration::nPdfAddrBits = nPdfAddrBits;
  OMTFConfiguration::nPdfValBits = nPdfValBits;
  OMTFConfiguration::nHitsPerLayer = nHitsPerLayer;
  OMTFConfiguration::nPhiBits = nPhiBits;
  OMTFConfiguration::nPhiBins = nPhiBins;
  OMTFConfiguration::nRefHits = nRefHits;
  OMTFConfiguration::nTestRefHits = nTestRefHits;
  OMTFConfiguration::nProcessors = nProcessors;
  OMTFConfiguration::nLogicRegions = nLogicRegions;
  OMTFConfiguration::nInputs = nInputs;
  OMTFConfiguration::nGoldenPatterns = nGoldenPatterns;

  ///Chamber sectors connections to logic processros.
  OMTFConfiguration::barrelMin =  std::vector<unsigned int>(6);
  OMTFConfiguration::barrelMax =  std::vector<unsigned int>(6);
  
  OMTFConfiguration::endcap10DegMin =  std::vector<unsigned int>(6);
  OMTFConfiguration::endcap10DegMax =  std::vector<unsigned int>(6);
  
  OMTFConfiguration::endcap20DegMin =  std::vector<unsigned int>(6);
  OMTFConfiguration::endcap20DegMax =  std::vector<unsigned int>(6);
  
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

    OMTFConfiguration::barrelMin[iProcessor] = barrelMin;
    OMTFConfiguration::endcap10DegMin[iProcessor] = endcap10DegMin;
    OMTFConfiguration::endcap20DegMin[iProcessor] = endcap20DegMin;

    OMTFConfiguration::barrelMax[iProcessor] = barrelMax;
    OMTFConfiguration::endcap10DegMax[iProcessor] = endcap10DegMax;
    OMTFConfiguration::endcap20DegMax[iProcessor] = endcap20DegMax;       
  }  
  
  ///hw <-> logic numbering map
  unsigned int nLogicLayers = 0;
  nElem = aOMTFElement->getElementsByTagName(_toDOMS("LayerMap"))->getLength();
  DOMElement* aLayerElement = 0;
  for(uint i=0;i<nElem;++i){
    aNode = aOMTFElement->getElementsByTagName(_toDOMS("LayerMap"))->item(i);
    aLayerElement = static_cast<DOMElement *>(aNode); 
    unsigned int hwNumber = std::atoi(_toString(aLayerElement->getAttribute(_toDOMS("hwNumber"))).c_str());
    unsigned int logicNumber = std::atoi(_toString(aLayerElement->getAttribute(_toDOMS("logicNumber"))).c_str());
    unsigned int isBendingLayer = std::atoi(_toString(aLayerElement->getAttribute(_toDOMS("bendingLayer"))).c_str());
    unsigned int iConnectedLayer = std::atoi(_toString(aLayerElement->getAttribute(_toDOMS("connectedToLayer"))).c_str());
    aConfig->hwToLogicLayer[hwNumber] = logicNumber;
    aConfig->logicToHwLayer[logicNumber] = hwNumber;    
    aConfig->logicToLogic[logicNumber] = iConnectedLayer;    
    if(isBendingLayer)     aConfig->bendingLayers.insert(logicNumber);    
    if(nLogicLayers<logicNumber) nLogicLayers = logicNumber;
  }
  ++nLogicLayers;//logic number in XML starts from 0.
  OMTFConfiguration::nLayers = nLogicLayers;

  ///ref<->logic numberig map
  unsigned int nRefLayers = 0;
  nElem = aOMTFElement->getElementsByTagName(_toDOMS("RefLayerMap"))->getLength();
  aConfig->refToLogicNumber.resize(nElem);
  DOMElement* aRefLayerElement = 0;
  for(uint i=0;i<nElem;++i){
    aNode = aOMTFElement->getElementsByTagName(_toDOMS("RefLayerMap"))->item(i);
    aRefLayerElement = static_cast<DOMElement *>(aNode); 
    unsigned int refLayer = std::atoi(_toString(aRefLayerElement->getAttribute(_toDOMS("refLayer"))).c_str());
    unsigned int logicNumber = std::atoi(_toString(aRefLayerElement->getAttribute(_toDOMS("logicNumber"))).c_str());
    aConfig->refToLogicNumber[refLayer] = logicNumber;
    if(nRefLayers<logicNumber) nRefLayers = refLayer;
  }
  ++nRefLayers;//ref number in XML starts from 0.
  OMTFConfiguration::nRefLayers = nRefLayers;

  ///processors initial phi for each reference layer
  std::vector<int> vector1D(OMTFConfiguration::nRefLayers,OMTFConfiguration::nPhiBins);
  OMTFConfiguration::processorPhiVsRefLayer.assign(OMTFConfiguration::nProcessors,vector1D);

  ///connections tables for each processor each logic cone
  ///Vector of all layers 
  OMTFConfiguration::vector1D_A aLayer1D(OMTFConfiguration::nLayers);
  ///Vector of all logic cones
  OMTFConfiguration::vector2D_A aLayer2D;
  aLayer2D.assign(OMTFConfiguration::nProcessors,aLayer1D);
  ///Vector of all processors
  OMTFConfiguration::connections.assign(OMTFConfiguration::nProcessors,aLayer2D);

  ///Starting phis of each region
  ///Vector of all regions in one processor
  std::vector<std::pair<int,int> > aRefHit1D(OMTFConfiguration::nLogicRegions,std::pair<int,int>(9999,9999));
  ///Vector of all reflayers
  std::vector<std::vector<std::pair<int,int> > > aRefHit2D;
  aRefHit2D.assign(OMTFConfiguration::nRefLayers,aRefHit1D);
  ///Vector of all processors
  OMTFConfiguration::regionPhisVsRefLayerVsProcessor.assign(OMTFConfiguration::nProcessors,aRefHit2D);

  //Vector of ref hit definitions
  std::vector<RefHitDef> aRefHitsDefs(OMTFConfiguration::nRefHits);
  ///Vector of all processros
  OMTFConfiguration::refHitsDefs.assign(OMTFConfiguration::nProcessors,aRefHitsDefs);

  nElem = aOMTFElement->getElementsByTagName(_toDOMS("Processor"))->getLength();
  assert(nElem==OMTFConfiguration::nProcessors);
  DOMElement* aProcessorElement = 0;
  for(uint i=0;i<nElem;++i){
    aNode = aOMTFElement->getElementsByTagName(_toDOMS("Processor"))->item(i);
    aProcessorElement = static_cast<DOMElement *>(aNode); 
    unsigned int iProcessor = std::atoi(_toString(aProcessorElement->getAttribute(_toDOMS("iProcessor"))).c_str());
    unsigned int nElem1 = aProcessorElement->getElementsByTagName(_toDOMS("RefLayer"))->getLength();
    assert(nElem1==OMTFConfiguration::nRefLayers);
    DOMElement* aRefLayerElement = 0;
    for(uint ii=0;ii<nElem1;++ii){
      aNode = aProcessorElement->getElementsByTagName(_toDOMS("RefLayer"))->item(ii);
      aRefLayerElement = static_cast<DOMElement *>(aNode); 
      unsigned int iRefLayer = std::atoi(_toString(aRefLayerElement->getAttribute(_toDOMS("iRefLayer"))).c_str());
      int iPhi = std::atoi(_toString(aRefLayerElement->getAttribute(_toDOMS("iGlobalPhiStart"))).c_str());
      OMTFConfiguration::processorPhiVsRefLayer[iProcessor][iRefLayer] = iPhi;
    }
    ///////////
    nElem1 = aProcessorElement->getElementsByTagName(_toDOMS("RefHit"))->getLength();
    assert(nElem1==OMTFConfiguration::nRefHits);
    DOMElement* aRefHitElement = 0;
    std::vector<int> starts;
    starts.assign(OMTFConfiguration::nRefLayers,-1);
    for(uint ii=0;ii<nElem1;++ii){
      aNode = aProcessorElement->getElementsByTagName(_toDOMS("RefHit"))->item(ii);
      aRefHitElement = static_cast<DOMElement *>(aNode); 
      unsigned int iRefHit = std::atoi(_toString(aRefHitElement->getAttribute(_toDOMS("iRefHit"))).c_str());
      int iPhiMin = std::atoi(_toString(aRefHitElement->getAttribute(_toDOMS("iPhiMin"))).c_str());
      int iPhiMax = std::atoi(_toString(aRefHitElement->getAttribute(_toDOMS("iPhiMax"))).c_str());
      unsigned int iInput = std::atoi(_toString(aRefHitElement->getAttribute(_toDOMS("iInput"))).c_str());
      unsigned int iRegion = std::atoi(_toString(aRefHitElement->getAttribute(_toDOMS("iRegion"))).c_str());
      unsigned int iRefLayer = std::atoi(_toString(aRefHitElement->getAttribute(_toDOMS("iRefLayer"))).c_str());
      /////////
      OMTFConfiguration::regionPhisVsRefLayerVsProcessor[iProcessor][iRefLayer][iRegion] = std::pair<int,int>(iPhiMin,iPhiMax);
      OMTFConfiguration::refHitsDefs[iProcessor][iRefHit] = RefHitDef(iInput,iPhiMin,iPhiMax,iRegion,iRefLayer);
    }
    ///////////
    unsigned int nElem2 = aProcessorElement->getElementsByTagName(_toDOMS("LogicRegion"))->getLength();
    assert(nElem2==OMTFConfiguration::nProcessors);
    DOMElement* aRegionElement = 0;
    for(uint ii=0;ii<nElem2;++ii){
      aNode = aProcessorElement->getElementsByTagName(_toDOMS("LogicRegion"))->item(ii);
      aRegionElement = static_cast<DOMElement *>(aNode); 
      unsigned int iRegion = std::atoi(_toString(aRegionElement->getAttribute(_toDOMS("iRegion"))).c_str());
      unsigned int nElem3 = aRegionElement->getElementsByTagName(_toDOMS("Layer"))->getLength();
      assert(nElem3==OMTFConfiguration::nLayers); 
      DOMElement* aLayerElement = 0;
      for(uint iii=0;iii<nElem3;++iii){
	aNode = aRegionElement->getElementsByTagName(_toDOMS("Layer"))->item(iii);
	aLayerElement = static_cast<DOMElement *>(aNode); 
	unsigned int iLayer = std::atoi(_toString(aLayerElement->getAttribute(_toDOMS("iLayer"))).c_str());
	unsigned int iFirstInput = std::atoi(_toString(aLayerElement->getAttribute(_toDOMS("iFirstInput"))).c_str());
	unsigned int nInputs = std::atoi(_toString(aLayerElement->getAttribute(_toDOMS("nInputs"))).c_str());
	OMTFConfiguration::connections[iProcessor][iRegion][iLayer] = std::pair<unsigned int, unsigned int>(iFirstInput,nInputs);
      }
    }   
  }
  delete doc;
}
//////////////////////////////////////////////////
//////////////////////////////////////////////////

