#include <iostream>
#include <cmath>
#include <algorithm>
#include <utility>
#include <array>

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
#include "Utilities/Xerces/interface/XercesStrUtils.h"

//////////////////////////////////
// XMLConfigReader
//////////////////////////////////

XMLConfigReader::XMLConfigReader()
{}

XMLConfigReader::~XMLConfigReader()
{}

//////////////////////////////////////////////////
//////////////////////////////////////////////////
void XMLConfigReader::readLUTs(std::vector<l1t::LUT*> luts,const L1TMuonOverlapParams & aConfig, const std::vector<std::string> & types){

  ///Fill payload string  
  auto const & aGPs = readPatterns(aConfig);

  for ( unsigned int i=0; i< luts.size(); i++ ) {
    l1t::LUT* lut=luts[i];
    const std::string &type=types[i];
    
    std::stringstream strStream;
    int totalInWidth = 7;//Number of bits used to address LUT
    int outWidth = 6;//Number of bits used to store LUT value
    
    if(type=="iCharge") outWidth = 1;
    if(type=="iEta") outWidth = 2;
    if(type=="iPt") outWidth = 9;
    if(type=="meanDistPhi"){
      outWidth = 11;
      totalInWidth = 14;
    }
    if(type=="pdf"){
      outWidth = 6;
      totalInWidth = 21;
    }
    
    ///Prepare the header 
    strStream <<"#<header> V1 "<<totalInWidth<<" "<<outWidth<<" </header> "<<std::endl;
    
    
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
}
//////////////////////////////////////////////////
//////////////////////////////////////////////////
unsigned int XMLConfigReader::getPatternsVersion() const{

  if(!patternsFile.size()) return 0;

  unsigned int version=0;
  XMLPlatformUtils::Initialize();
  {
    XercesDOMParser parser;
    parser.setValidationScheme(XercesDOMParser::Val_Auto);
    parser.setDoNamespaces(false);
    
    parser.parse(patternsFile.c_str()); 
    xercesc::DOMDocument* doc = parser.getDocument();
    assert(doc);

    DOMNode *aNode = doc->getElementsByTagName(cms::xerces::uStr("OMTF").ptr())->item(0);
    DOMElement* aOMTFElement = static_cast<DOMElement *>(aNode);
    
    version = std::stoul(cms::xerces::toString(aOMTFElement->getAttribute(cms::xerces::uStr("version").ptr())), nullptr, 16);
    parser.resetDocumentPool();
  }
  XMLPlatformUtils::Terminate();
  
  return version;
}

//////////////////////////////////////////////////
//////////////////////////////////////////////////
std::vector<std::shared_ptr<GoldenPattern>> XMLConfigReader::readPatterns(const L1TMuonOverlapParams & aConfig){

  aGPs.clear();
  
  XMLPlatformUtils::Initialize();

  auto const &xmlGP = cms::xerces::uStr("GP");
  std::array<cms::xerces::ZStr<short unsigned int>, 4> xmliPt= {{cms::xerces::uStr("iPt1"),cms::xerces::uStr("iPt2"),cms::xerces::uStr("iPt3"),cms::xerces::uStr("iPt4") }};

  {
    XercesDOMParser parser;
    parser.setValidationScheme(XercesDOMParser::Val_Auto);
    parser.setDoNamespaces(false);
    
    parser.parse(patternsFile.c_str()); 
    xercesc::DOMDocument* doc = parser.getDocument();
    assert(doc);
    
    unsigned int nElem = doc->getElementsByTagName(xmlGP.ptr())->getLength();
    if(nElem<1){
      edm::LogError("critical")<<"Problem parsing XML file "<<patternsFile<<std::endl;
      edm::LogError("critical")<<"No GoldenPattern items: GP found"<<std::endl;
      return aGPs;
    }
    
    DOMNode *aNode = 0;
    DOMElement* aGPElement = 0;
    unsigned int iGPNumber=0;
    
    for(unsigned int iItem=0;iItem<nElem;++iItem){
      aNode = doc->getElementsByTagName(xmlGP.ptr())->item(iItem);
      aGPElement = static_cast<DOMElement *>(aNode);
      
      std::unique_ptr<GoldenPattern> aGP;
      for(unsigned int index = 1;index<5;++index){
	///Patterns XML format backward compatibility. Can use both packed by 4, or by 1 XML files.      
	if(aGPElement->getAttributeNode(xmliPt[index-1].ptr())) {
	  aGP = buildGP(aGPElement, aConfig, index, iGPNumber);
	  if(aGP){	  
	    aGPs.emplace_back(std::move(aGP));
	    iGPNumber++;
	  }
	}
	else{
	  aGP = buildGP(aGPElement, aConfig);
	  if(aGP){
	    aGPs.emplace_back(std::move(aGP));
	    iGPNumber++;
	  }
	  break;
	}
      }
    }
    
    // Reset the documents vector pool and release all the associated memory back to the system.
    parser.resetDocumentPool();
  }

  XMLPlatformUtils::Terminate();

  return aGPs;
}
//////////////////////////////////////////////////
//////////////////////////////////////////////////
std::unique_ptr<GoldenPattern> XMLConfigReader::buildGP(DOMElement* aGPElement,
					 const L1TMuonOverlapParams & aConfig,
					 unsigned int index,
					 unsigned int aGPNumber){


  auto const &xmliEta = cms::xerces::uStr("iEta");
  //index 0 means no number at the end
  std::ostringstream stringStr;
  if (index>0) stringStr<<"iPt"<<index;
  else stringStr.str("iPt");
  auto const &xmliPt = cms::xerces::uStr(stringStr.str().c_str());
  stringStr.str("");
  if (index>0) stringStr<<"value"<<index;
  else stringStr.str("value");
  auto const &xmlValue = cms::xerces::uStr(stringStr.str().c_str());
  
  auto const &xmliCharge = cms::xerces::uStr("iCharge");
  auto const &xmlLayer = cms::xerces::uStr("Layer");
  auto const &xmlRefLayer = cms::xerces::uStr("RefLayer");
  auto const &xmlmeanDistPhi = cms::xerces::uStr("meanDistPhi");
  auto const &xmlPDF = cms::xerces::uStr("PDF");
  
  unsigned int iPt = cms::xerces::toUInt(aGPElement->getAttribute(xmliPt.ptr()));  
  int iEta = cms::xerces::toUInt(aGPElement->getAttribute(xmliEta.ptr()));
  int iCharge = cms::xerces::toUInt(aGPElement->getAttribute(xmliCharge.ptr()));
  int val = 0;
  unsigned int nLayers = aGPElement->getElementsByTagName(xmlLayer.ptr())->getLength();
  assert(nLayers==(unsigned) aConfig.nLayers());

  DOMNode *aNode = 0;
  DOMElement* aLayerElement = 0;
  DOMElement* aItemElement = 0;
  GoldenPattern::vector2D meanDistPhi2D(nLayers);
  GoldenPattern::vector1D pdf1D(exp2(aConfig.nPdfAddrBits()));
  GoldenPattern::vector3D pdf3D(aConfig.nLayers());
  GoldenPattern::vector2D pdf2D(aConfig.nRefLayers());

  if(iPt==0){///Build empty GP
    GoldenPattern::vector1D meanDistPhi1D(aConfig.nRefLayers());
    meanDistPhi2D.assign(aConfig.nLayers(),meanDistPhi1D);
    pdf1D.assign(exp2(aConfig.nPdfAddrBits()),0);
    pdf2D.assign(aConfig.nRefLayers(),pdf1D);
    pdf3D.assign(aConfig.nLayers(),pdf2D);

    Key aKey(iEta,iPt,iCharge, aGPNumber);
    auto aGP = std::make_unique<GoldenPattern>(aKey,static_cast<const OMTFConfiguration*>(nullptr));
    aGP->setMeanDistPhi(meanDistPhi2D);
    aGP->setPdf(pdf3D);
    return aGP;
  }
  
  ///Loop over layers
  for(unsigned int iLayer=0;iLayer<nLayers;++iLayer){
    aNode = aGPElement->getElementsByTagName(xmlLayer.ptr())->item(iLayer);
    aLayerElement = static_cast<DOMElement *>(aNode); 
    ///MeanDistPhi vector
    unsigned int nItems = aLayerElement->getElementsByTagName(xmlRefLayer.ptr())->getLength();
    assert(nItems==(unsigned) aConfig.nRefLayers());
    GoldenPattern::vector1D meanDistPhi1D(nItems);
    for(unsigned int iItem=0;iItem<nItems;++iItem){
      aNode = aLayerElement->getElementsByTagName(xmlRefLayer.ptr())->item(iItem);
      aItemElement = static_cast<DOMElement *>(aNode); 
      val = cms::xerces::toUInt(aItemElement->getAttribute(xmlmeanDistPhi.ptr()));
      meanDistPhi1D[iItem] = val;
    }
    meanDistPhi2D[iLayer] = meanDistPhi1D;

    ///PDF vector
    nItems = aLayerElement->getElementsByTagName(xmlPDF.ptr())->getLength();
    assert(nItems==aConfig.nRefLayers()*exp2(aConfig.nPdfAddrBits()));
    for(unsigned int iRefLayer=0;iRefLayer<(unsigned) aConfig.nRefLayers();++iRefLayer){
      pdf1D.assign(exp2(aConfig.nPdfAddrBits()),0);
      for(unsigned int iPdf=0;iPdf<exp2(aConfig.nPdfAddrBits());++iPdf){
	aNode = aLayerElement->getElementsByTagName(xmlPDF.ptr())->item(iRefLayer*exp2(aConfig.nPdfAddrBits())+iPdf);
	aItemElement = static_cast<DOMElement *>(aNode);
	val = cms::xerces::toUInt(aItemElement->getAttribute(xmlValue.ptr()));
	pdf1D[iPdf] = val;
      }
      pdf2D[iRefLayer] = pdf1D;
    }
    pdf3D[iLayer] = pdf2D;
  }

  Key aKey(iEta,iPt,iCharge, aGPNumber);
  auto aGP = std::make_unique<GoldenPattern>(aKey,static_cast<const OMTFConfiguration*>(nullptr));
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
 
}
//////////////////////////////////////////////////
//////////////////////////////////////////////////
void XMLConfigReader::readConfig(L1TMuonOverlapParams *aConfig) const{

  XMLPlatformUtils::Initialize();
  {
    XercesDOMParser parser;
    parser.setValidationScheme(XercesDOMParser::Val_Auto);
    parser.setDoNamespaces(false);

    auto const &xmlOMTF= cms::xerces::uStr("OMTF");
    auto const &xmlversion= cms::xerces::uStr("version");
    auto const &xmlGlobalData= cms::xerces::uStr("GlobalData");
    auto const &xmlnPdfAddrBits= cms::xerces::uStr("nPdfAddrBits");
    auto const &xmlnPdfValBits= cms::xerces::uStr("nPdfValBits");
    auto const &xmlnPhiBits= cms::xerces::uStr("nPhiBits");
    auto const &xmlnPhiBins= cms::xerces::uStr("nPhiBins");
    auto const &xmlnProcessors = cms::xerces::uStr("nProcessors");
    auto const &xmlnLogicRegions = cms::xerces::uStr("nLogicRegions");
    auto const &xmlnInputs= cms::xerces::uStr("nInputs");
    auto const &xmlnLayers= cms::xerces::uStr("nLayers");
    auto const &xmlnRefLayers= cms::xerces::uStr("nRefLayers");
    auto const &xmliProcessor= cms::xerces::uStr("iProcessor");
    auto const &xmlbarrelMin= cms::xerces::uStr("barrelMin");
    auto const &xmlbarrelMax= cms::xerces::uStr("barrelMax");
    auto const &xmlendcap10DegMin= cms::xerces::uStr("endcap10DegMin");
    auto const &xmlendcap10DegMax= cms::xerces::uStr("endcap10DegMax");
    auto const &xmlendcap20DegMin= cms::xerces::uStr("endcap20DegMin");
    auto const &xmlendcap20DegMax= cms::xerces::uStr("endcap20DegMax");
    auto const &xmlLayerMap = cms::xerces::uStr("LayerMap");
    auto const &xmlhwNumber = cms::xerces::uStr("hwNumber");
    auto const &xmllogicNumber = cms::xerces::uStr("logicNumber");
    auto const &xmlbendingLayer = cms::xerces::uStr("bendingLayer");
    auto const &xmlconnectedToLayer = cms::xerces::uStr("connectedToLayer");
    auto const &xmlRefLayerMap = cms::xerces::uStr("RefLayerMap");
    auto const &xmlrefLayer = cms::xerces::uStr("refLayer");
    auto const &xmlProcessor = cms::xerces::uStr("Processor");
    auto const &xmlRefLayer = cms::xerces::uStr("RefLayer");
    auto const &xmliRefLayer = cms::xerces::uStr("iRefLayer");
    auto const &xmliGlobalPhiStart = cms::xerces::uStr("iGlobalPhiStart");
    auto const &xmlRefHit = cms::xerces::uStr("RefHit");
    auto const &xmliRefHit = cms::xerces::uStr("iRefHit");
    auto const &xmliPhiMin = cms::xerces::uStr("iPhiMin");
    auto const &xmliPhiMax = cms::xerces::uStr("iPhiMax");
    auto const &xmliInput = cms::xerces::uStr("iInput");
    auto const &xmliRegion = cms::xerces::uStr("iRegion");
    auto const &xmlLogicRegion = cms::xerces::uStr("LogicRegion");
    auto const &xmlLayer = cms::xerces::uStr("Layer");
    auto const &xmliLayer = cms::xerces::uStr("iLayer");
    auto const &xmliFirstInput = cms::xerces::uStr("iFirstInput");
    auto const &xmlnHitsPerLayer = cms::xerces::uStr("nHitsPerLayer");
    auto const &xmlnRefHits = cms::xerces::uStr("nRefHits");
    auto const &xmlnTestRefHits = cms::xerces::uStr("nTestRefHits");
    auto const &xmlnGoldenPatterns = cms::xerces::uStr("nGoldenPatterns");
    auto const &xmlConnectionMap = cms::xerces::uStr("ConnectionMap");
    parser.parse(configFile.c_str()); 
    xercesc::DOMDocument* doc = parser.getDocument();
    assert(doc);
    unsigned int nElem = doc->getElementsByTagName(xmlOMTF.ptr())->getLength();
    if(nElem!=1){
      edm::LogError("critical")<<"Problem parsing XML file "<<configFile<<std::endl;
      assert(nElem==1);
    }
    DOMNode *aNode = doc->getElementsByTagName(xmlOMTF.ptr())->item(0);
    DOMElement* aOMTFElement = static_cast<DOMElement *>(aNode);
    
    unsigned int version = std::stoul(cms::xerces::toString(aOMTFElement->getAttribute(xmlversion.ptr())), nullptr, 16);
    aConfig->setFwVersion(version);
    
    ///Addresing bits numbers
    nElem = aOMTFElement->getElementsByTagName(xmlGlobalData.ptr())->getLength();
    assert(nElem==1);
    aNode = aOMTFElement->getElementsByTagName(xmlGlobalData.ptr())->item(0);
    DOMElement* aElement = static_cast<DOMElement *>(aNode); 
    
    unsigned int nPdfAddrBits = cms::xerces::toUInt(aElement->getAttribute(xmlnPdfAddrBits.ptr())); 
    unsigned int nPdfValBits = cms::xerces::toUInt(aElement->getAttribute(xmlnPdfValBits.ptr())); 
    unsigned int nHitsPerLayer = cms::xerces::toUInt(aElement->getAttribute(xmlnHitsPerLayer.ptr())); 
    unsigned int nPhiBits = cms::xerces::toUInt(aElement->getAttribute(xmlnPhiBits.ptr())); 
    unsigned int nPhiBins = cms::xerces::toUInt(aElement->getAttribute(xmlnPhiBins.ptr())); 

    unsigned int nRefHits = cms::xerces::toUInt(aElement->getAttribute(xmlnRefHits.ptr())); 
    unsigned int nTestRefHits = cms::xerces::toUInt(aElement->getAttribute(xmlnTestRefHits.ptr()));
    unsigned int nProcessors = cms::xerces::toUInt(aElement->getAttribute(xmlnProcessors.ptr()));
    unsigned int nLogicRegions = cms::xerces::toUInt(aElement->getAttribute(xmlnLogicRegions.ptr()));
    unsigned int nInputs = cms::xerces::toUInt(aElement->getAttribute(xmlnInputs.ptr()));
    unsigned int nLayers = cms::xerces::toUInt(aElement->getAttribute(xmlnLayers.ptr()));
    unsigned int nRefLayers = cms::xerces::toUInt(aElement->getAttribute(xmlnRefLayers.ptr()));
    unsigned int nGoldenPatterns = cms::xerces::toUInt(aElement->getAttribute(xmlnGoldenPatterns.ptr()));
    
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
    nElem = aOMTFElement->getElementsByTagName(xmlConnectionMap.ptr())->getLength();
    DOMElement* aConnectionElement = 0;
    for(unsigned int i=0;i<nElem;++i){
      aNode = aOMTFElement->getElementsByTagName(xmlConnectionMap.ptr())->item(i);
      aConnectionElement = static_cast<DOMElement *>(aNode);
      unsigned int iProcessor = cms::xerces::toUInt(aConnectionElement->getAttribute(xmliProcessor.ptr()));
      unsigned int barrelMin = cms::xerces::toUInt(aConnectionElement->getAttribute(xmlbarrelMin.ptr()));
      unsigned int barrelMax = cms::xerces::toUInt(aConnectionElement->getAttribute(xmlbarrelMax.ptr()));
      unsigned int endcap10DegMin = cms::xerces::toUInt(aConnectionElement->getAttribute(xmlendcap10DegMin.ptr()));
      unsigned int endcap10DegMax = cms::xerces::toUInt(aConnectionElement->getAttribute(xmlendcap10DegMax.ptr()));
      unsigned int endcap20DegMin = cms::xerces::toUInt(aConnectionElement->getAttribute(xmlendcap20DegMin.ptr()));
      unsigned int endcap20DegMax = cms::xerces::toUInt(aConnectionElement->getAttribute(xmlendcap20DegMax.ptr()));
      
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
    
    nElem = aOMTFElement->getElementsByTagName(xmlLayerMap.ptr())->getLength();
    DOMElement* aLayerElement = 0;
    for(unsigned int i=0;i<nElem;++i){
      aNode = aOMTFElement->getElementsByTagName(xmlLayerMap.ptr())->item(i);
      aLayerElement = static_cast<DOMElement *>(aNode); 
      unsigned int hwNumber = cms::xerces::toUInt(aLayerElement->getAttribute(xmlhwNumber.ptr()));
      unsigned int logicNumber = cms::xerces::toUInt(aLayerElement->getAttribute(xmllogicNumber.ptr()));
      unsigned int isBendingLayer = cms::xerces::toUInt(aLayerElement->getAttribute(xmlbendingLayer.ptr()));
      unsigned int iConnectedLayer = cms::xerces::toUInt(aLayerElement->getAttribute(xmlconnectedToLayer.ptr()));
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
    
    nElem = aOMTFElement->getElementsByTagName(xmlRefLayerMap.ptr())->getLength();
    DOMElement* aRefLayerElement = 0;
    for(unsigned int i=0;i<nElem;++i){
      aNode = aOMTFElement->getElementsByTagName(xmlRefLayerMap.ptr())->item(i);
      aRefLayerElement = static_cast<DOMElement *>(aNode); 
      unsigned int refLayer = cms::xerces::toUInt(aRefLayerElement->getAttribute(xmlrefLayer.ptr()));
      unsigned int logicNumber = cms::xerces::toUInt(aRefLayerElement->getAttribute(xmllogicNumber.ptr()));
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
    
    nElem = aOMTFElement->getElementsByTagName(xmlProcessor.ptr())->getLength();
    assert(nElem==nProcessors);
    DOMElement* aProcessorElement = 0;
    for(unsigned int i=0;i<nElem;++i){
      aNode = aOMTFElement->getElementsByTagName(xmlProcessor.ptr())->item(i);
      aProcessorElement = static_cast<DOMElement *>(aNode); 
      unsigned int iProcessor = cms::xerces::toUInt(aProcessorElement->getAttribute(xmliProcessor.ptr()));
      unsigned int nElem1 = aProcessorElement->getElementsByTagName(xmlRefLayer.ptr())->getLength();
      assert(nElem1==nRefLayers);
      DOMElement* aRefLayerElement = 0;
      for(unsigned int ii=0;ii<nElem1;++ii){
	aNode = aProcessorElement->getElementsByTagName(xmlRefLayer.ptr())->item(ii);
	aRefLayerElement = static_cast<DOMElement *>(aNode); 
	unsigned int iRefLayer = cms::xerces::toUInt(aRefLayerElement->getAttribute(xmliRefLayer.ptr()));
	int iPhi = cms::xerces::toUInt(aRefLayerElement->getAttribute(xmliGlobalPhiStart.ptr()));
	aGlobalPhiStartVec[iRefLayer + iProcessor*nRefLayers] = iPhi;
      }
      ///////////
      nElem1 = aProcessorElement->getElementsByTagName(xmlRefHit.ptr())->getLength();
      assert( (iProcessor==0 && nElem1==nRefHits) || (iProcessor!=0 && nElem1==0) );
      DOMElement* aRefHitElement = 0;
      for(unsigned int ii=0;ii<nElem1;++ii){
	aNode = aProcessorElement->getElementsByTagName(xmlRefHit.ptr())->item(ii);
	aRefHitElement = static_cast<DOMElement *>(aNode); 
	unsigned int iRefHit = cms::xerces::toUInt(aRefHitElement->getAttribute(xmliRefHit.ptr()));
	int iPhiMin = cms::xerces::toUInt(aRefHitElement->getAttribute(xmliPhiMin.ptr()));
	int iPhiMax = cms::xerces::toUInt(aRefHitElement->getAttribute(xmliPhiMax.ptr()));
	unsigned int iInput = cms::xerces::toUInt(aRefHitElement->getAttribute(xmliInput.ptr()));
	unsigned int iRegion = cms::xerces::toUInt(aRefHitElement->getAttribute(xmliRegion.ptr()));
	unsigned int iRefLayer = cms::xerces::toUInt(aRefHitElement->getAttribute(xmliRefLayer.ptr()));
	
	aRefHitNode.iRefHit = iRefHit;
	aRefHitNode.iPhiMin = iPhiMin;
	aRefHitNode.iPhiMax = iPhiMax;
	aRefHitNode.iInput = iInput;
	aRefHitNode.iRegion = iRegion;
	aRefHitNode.iRefLayer = iRefLayer;
	for (unsigned int iProcessor=0; iProcessor<nProcessors; iProcessor++) aRefHitMapVec[iRefHit + iProcessor*nRefHits] = aRefHitNode;
      }
      ///////////
      unsigned int nElem2 = aProcessorElement->getElementsByTagName(xmlLogicRegion.ptr())->getLength();
      assert( (iProcessor==0 && nElem2==nLogicRegions) || (iProcessor!=0 && nElem2==0) );
      DOMElement* aRegionElement = 0;
      for(unsigned int ii=0;ii<nElem2;++ii){
	aNode = aProcessorElement->getElementsByTagName(xmlLogicRegion.ptr())->item(ii);
	aRegionElement = static_cast<DOMElement *>(aNode); 
	unsigned int iRegion = cms::xerces::toUInt(aRegionElement->getAttribute(xmliRegion.ptr()));
	unsigned int nElem3 = aRegionElement->getElementsByTagName(xmlLayer.ptr())->getLength();
	assert(nElem3==nLayers);
	DOMElement* aLayerElement = 0;
	for(unsigned int iii=0;iii<nElem3;++iii){
  	  aNode = aRegionElement->getElementsByTagName(xmlLayer.ptr())->item(iii);
	  aLayerElement = static_cast<DOMElement *>(aNode); 
	  unsigned int iLayer = cms::xerces::toUInt(aLayerElement->getAttribute(xmliLayer.ptr()));
	  unsigned int iFirstInput = cms::xerces::toUInt(aLayerElement->getAttribute(xmliFirstInput.ptr()));
	  unsigned int nInputs = cms::xerces::toUInt(aLayerElement->getAttribute(xmlnInputs.ptr()));
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
    
    // Reset the documents vector pool and release all the associated memory back to the system.
    parser.resetDocumentPool();
  }
  XMLPlatformUtils::Terminate();
}
