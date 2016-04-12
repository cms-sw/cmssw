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
XMLConfigReader::XMLConfigReader(OMTFConfiguration * omtf_config) : m_omtf_config(omtf_config) {

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
  const std::vector<GoldenPattern *> & aGPs = readPatterns();
  unsigned int in = 0;
  int out = 0;
  for(auto it: aGPs){
    if(type=="iCharge") out = it->key().theCharge==-1 ? 0:1;
    if(type=="iEta") out = it->key().theEtaCode;
    if(type=="iPt") out = it->key().thePtCode;
    if(type=="meanDistPhi"){
      for(unsigned int iLayer = 0;iLayer<m_omtf_config->nLayers;++iLayer){
	for(unsigned int iRefLayer=0;iRefLayer<m_omtf_config->nRefLayers;++iRefLayer){
	  out = (1<<(outWidth-1)) + it->meanDistPhiValue(iLayer,iRefLayer);
	  strStream<<in<<" "<<out<<std::endl;
	  ++in;
	}
      }
    }
    if(type=="pdf"){
      for(unsigned int iLayer = 0;iLayer<m_omtf_config->nLayers;++iLayer){
	for(unsigned int iRefLayer=0;iRefLayer<m_omtf_config->nRefLayers;++iRefLayer){
	  for(unsigned int iPdf=0;iPdf<exp2(m_omtf_config->nPdfAddrBits);++iPdf){
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

  //if(aGPs.size()) return aGPs;
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
      ///Patterns XMl format backward compatibility. Can use both packed by 4, or by 1 XML files.      
      if(aGPElement->getAttributeNode(_toDOMS(stringStr.str().c_str()))){
	aGP = buildGP(aGPElement,index);
	if(aGP) aGPs.push_back(aGP);
      }
      else{
	aGP = buildGP(aGPElement);
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
  assert(nLayers==m_omtf_config->nLayers);
  DOMNode *aNode = 0;
  DOMElement* aLayerElement = 0;
  DOMElement* aItemElement = 0;
  GoldenPattern::vector2D meanDistPhi2D(nLayers);
  GoldenPattern::vector1D pdf1D(exp2(m_omtf_config->nPdfAddrBits));
  GoldenPattern::vector3D pdf3D(m_omtf_config->nLayers);
  GoldenPattern::vector2D pdf2D(m_omtf_config->nRefLayers);
  ///Loop over layers
  for(unsigned int iLayer=0;iLayer<nLayers;++iLayer){
    aNode = aGPElement->getElementsByTagName(_toDOMS("Layer"))->item(iLayer);
    aLayerElement = static_cast<DOMElement *>(aNode); 
    ///MeanDistPhi vector
    unsigned int nItems = aLayerElement->getElementsByTagName(_toDOMS("RefLayer"))->getLength();
    assert(nItems==m_omtf_config->nRefLayers);
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
    assert(nItems==m_omtf_config->nRefLayers*exp2(m_omtf_config->nPdfAddrBits));
    for(unsigned int iRefLayer=0;iRefLayer<m_omtf_config->nRefLayers;++iRefLayer){
      pdf1D.assign(exp2(m_omtf_config->nPdfAddrBits),0);
      for(unsigned int iPdf=0;iPdf<exp2(m_omtf_config->nPdfAddrBits);++iPdf){
	aNode = aLayerElement->getElementsByTagName(_toDOMS("PDF"))->item(iRefLayer*exp2(m_omtf_config->nPdfAddrBits)+iPdf);
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
      //aRefHitMapVec[iRefHit + iProcessor*nRefHits] = aRefHitNode;
      for (unsigned int ip=0; ip<nProcessors; ++ip) aRefHitMapVec[iRefHit + ip*nRefHits] = aRefHitNode;
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
//        aLayerInputMapVec[iLayer + iRegion*nLayers + iProcessor*nLayers*nLogicRegions] = aLayerInputNode;
	  for (unsigned int ip=0; ip<nProcessors; ++ip) aLayerInputMapVec[iLayer + iRegion*nLayers + ip*nLayers*nLogicRegions] = aLayerInputNode;
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
  unsigned int fwVersion = std::atoi(_toString(aOMTFElement->getAttribute(_toDOMS("version"))).c_str()); 

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
  m_omtf_config->fwVersion = fwVersion;
  m_omtf_config->minPdfVal = minPdfVal;
  m_omtf_config->nPdfAddrBits = nPdfAddrBits;
  m_omtf_config->nPdfValBits = nPdfValBits;
  m_omtf_config->nHitsPerLayer = nHitsPerLayer;
  m_omtf_config->nPhiBits = nPhiBits;
  m_omtf_config->nPhiBins = nPhiBins;
  m_omtf_config->nRefHits = nRefHits;
  m_omtf_config->nTestRefHits = nTestRefHits;
  m_omtf_config->nProcessors = nProcessors;
  m_omtf_config->nLogicRegions = nLogicRegions;
  m_omtf_config->nInputs = nInputs;
  m_omtf_config->nGoldenPatterns = nGoldenPatterns;

  ///Chamber sectors connections to logic processros.
  m_omtf_config->barrelMin =  std::vector<unsigned int>(6);
  m_omtf_config->barrelMax =  std::vector<unsigned int>(6);
  
  m_omtf_config->endcap10DegMin =  std::vector<unsigned int>(6);
  m_omtf_config->endcap10DegMax =  std::vector<unsigned int>(6);
  
  m_omtf_config->endcap20DegMin =  std::vector<unsigned int>(6);
  m_omtf_config->endcap20DegMax =  std::vector<unsigned int>(6);
  
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

    m_omtf_config->barrelMin[iProcessor] = barrelMin;
    m_omtf_config->endcap10DegMin[iProcessor] = endcap10DegMin;
    m_omtf_config->endcap20DegMin[iProcessor] = endcap20DegMin;

    m_omtf_config->barrelMax[iProcessor] = barrelMax;
    m_omtf_config->endcap10DegMax[iProcessor] = endcap10DegMax;
    m_omtf_config->endcap20DegMax[iProcessor] = endcap20DegMax;       
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
  m_omtf_config->nLayers = nLogicLayers;

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
  m_omtf_config->nRefLayers = nRefLayers;

  ///processors initial phi for each reference layer
  std::vector<int> vector1D(m_omtf_config->nRefLayers,m_omtf_config->nPhiBins);
  m_omtf_config->processorPhiVsRefLayer.assign(m_omtf_config->nProcessors,vector1D);

  ///connections tables for each processor each logic cone
  ///Vector of all layers 
  OMTFConfiguration::vector1D_A aLayer1D(m_omtf_config->nLayers);
  ///Vector of all logic regions
  OMTFConfiguration::vector2D_A aLayer2D;
  aLayer2D.assign(m_omtf_config->nLogicRegions,aLayer1D);
  ///Vector of all processors
  m_omtf_config->connections.assign(m_omtf_config->nProcessors,aLayer2D);

  ///Starting phis of each region
  ///Vector of all regions in one processor
  std::vector<std::pair<int,int> > aRefHit1D(m_omtf_config->nLogicRegions,std::pair<int,int>(9999,9999));
  ///Vector of all reflayers
  std::vector<std::vector<std::pair<int,int> > > aRefHit2D;
  aRefHit2D.assign(m_omtf_config->nRefLayers,aRefHit1D);
  ///Vector of all inputs
  m_omtf_config->regionPhisVsRefLayerVsInput.assign(m_omtf_config->nInputs,aRefHit2D);

  //Vector of ref hit definitions
  std::vector<RefHitDef> aRefHitsDefs(m_omtf_config->nRefHits);
  ///Vector of all processros
  m_omtf_config->refHitsDefs.assign(m_omtf_config->nProcessors,aRefHitsDefs);

  nElem = aOMTFElement->getElementsByTagName(_toDOMS("Processor"))->getLength();
  assert(nElem==m_omtf_config->nProcessors);
  DOMElement* aProcessorElement = 0;
  for(uint i=0;i<nElem;++i){
    aNode = aOMTFElement->getElementsByTagName(_toDOMS("Processor"))->item(i);
    aProcessorElement = static_cast<DOMElement *>(aNode); 
    unsigned int iProcessor = std::atoi(_toString(aProcessorElement->getAttribute(_toDOMS("iProcessor"))).c_str());
    unsigned int nElem1 = aProcessorElement->getElementsByTagName(_toDOMS("RefLayer"))->getLength();
    assert(nElem1==m_omtf_config->nRefLayers);
    DOMElement* aRefLayerElement = 0;
    for(uint ii=0;ii<nElem1;++ii){
      aNode = aProcessorElement->getElementsByTagName(_toDOMS("RefLayer"))->item(ii);
      aRefLayerElement = static_cast<DOMElement *>(aNode); 
      unsigned int iRefLayer = std::atoi(_toString(aRefLayerElement->getAttribute(_toDOMS("iRefLayer"))).c_str());
      int iPhi = std::atoi(_toString(aRefLayerElement->getAttribute(_toDOMS("iGlobalPhiStart"))).c_str());
      m_omtf_config->processorPhiVsRefLayer[iProcessor][iRefLayer] = iPhi;      
    }
    ///////////
    nElem1 = aProcessorElement->getElementsByTagName(_toDOMS("RefHit"))->getLength();    
    assert((iProcessor==0 && nElem1==nRefHits) || (iProcessor!=0 && nElem1==0) );
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
      /////////
      m_omtf_config->regionPhisVsRefLayerVsInput[iInput][iRefLayer][iRegion] = std::pair<int,int>(iPhiMin,iPhiMax);
      m_omtf_config->refHitsDefs[iProcessor][iRefHit] = RefHitDef(iInput,iPhiMin,iPhiMax,iRegion,iRefLayer);
      ///Fill all processors with the same setting as for processor 0.
      if(iProcessor==0){
	for (unsigned int iProcessorTmp=0; iProcessorTmp<m_omtf_config->nProcessors; ++iProcessorTmp){
	  m_omtf_config->refHitsDefs[iProcessorTmp][iRefHit] = RefHitDef(iInput,iPhiMin,iPhiMax,iRegion,iRefLayer);
	}      
      }
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
      assert(nElem3==m_omtf_config->nLayers); 
      DOMElement* aLayerElement = 0;
      for(uint iii=0;iii<nElem3;++iii){
	aNode = aRegionElement->getElementsByTagName(_toDOMS("Layer"))->item(iii);
	aLayerElement = static_cast<DOMElement *>(aNode); 
	unsigned int iLayer = std::atoi(_toString(aLayerElement->getAttribute(_toDOMS("iLayer"))).c_str());
	unsigned int iFirstInput = std::atoi(_toString(aLayerElement->getAttribute(_toDOMS("iFirstInput"))).c_str());
	unsigned int nInputs = std::atoi(_toString(aLayerElement->getAttribute(_toDOMS("nInputs"))).c_str());
	m_omtf_config->connections[iProcessor][iRegion][iLayer] = std::pair<unsigned int, unsigned int>(iFirstInput,nInputs);
	///Fill all processors with the same setting as for processor 0.
	if(iProcessor==0){
	  for (unsigned int iProcessorTmp=0; iProcessorTmp<m_omtf_config->nProcessors; ++iProcessorTmp){
	    m_omtf_config->connections[iProcessorTmp][iRegion][iLayer] = std::pair<unsigned int, unsigned int>(iFirstInput,nInputs);
	  }      
	}
      }
    }   
  }
  delete doc;
}
//////////////////////////////////////////////////
//////////////////////////////////////////////////

