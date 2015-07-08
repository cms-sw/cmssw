#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <xercesc/dom/DOMNode.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include "FWCore/Concurrency/interface/Xerces.h"
#include <xercesc/util/XMLString.hpp>
#include <xercesc/sax/SAXException.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <TString.h>

#include "CondFormats/EcalObjects/interface/EcalPulseSymmCovariances.h"
#include "CondTools/Ecal/interface/EcalPulseSymmCovariancesXMLTranslator.h"
#include "CondTools/Ecal/interface/DOMHelperFunctions.h"
#include "CondTools/Ecal/interface/XMLTags.h"


using namespace XERCES_CPP_NAMESPACE;
using namespace xuti;
using namespace std;

int  EcalPulseSymmCovariancesXMLTranslator::readXML(const std::string& filename, 
					  EcalCondHeader& header,
					  EcalPulseSymmCovariances& record){

  cms::concurrency::xercesInitialize();

  XercesDOMParser* parser = new XercesDOMParser;
  parser->setValidationScheme( XercesDOMParser::Val_Never );
  parser->setDoNamespaces( false );
  parser->setDoSchema( false );
  
  parser->parse(filename.c_str());

  DOMDocument* xmlDoc = parser->getDocument();
  if (!xmlDoc) {
    std::cout << "EcalPulseSymmCovariancesXMLTranslator::Error parsing document" << std::endl;
    return -1;
  }

  DOMElement* elementRoot = xmlDoc->getDocumentElement();

  xuti::readHeader(elementRoot,header);

  DOMNode * cellnode = getChildNode(elementRoot,Cell_tag);

  int chan = 0;
  while(cellnode) {
    //    std::cout << " Channel " << chan << std::endl;
    float covs[EcalPulseShape::TEMPLATESAMPLES*(EcalPulseShape::TEMPLATESAMPLES+1)/2];

    DetId detid = readCellId(dynamic_cast<DOMElement*>(cellnode));
    //    std::cout << " readCell Id Channel " << chan << " tag " << mean12_tag << std::endl;

    std::vector<std::string> pulsecov_tag(static_cast<size_t>(EcalPulseShape::TEMPLATESAMPLES*(EcalPulseShape::TEMPLATESAMPLES+1)/2));
    for(int i=0; i<EcalPulseShape::TEMPLATESAMPLES; ++i) { 
      for(int j=0; j<EcalPulseShape::TEMPLATESAMPLES; ++j) {
        int k=-1;
        if(j >= i) k = j + (EcalPulseShape::TEMPLATESAMPLES-1)*i;
        else k = i + (EcalPulseShape::TEMPLATESAMPLES-1)*j;
        pulsecov_tag[k] = Form("samplecov_%d_%d",i,j);
      }
    }

    DOMNode** covs_node = new DOMNode*[int(EcalPulseShape::TEMPLATESAMPLES*(EcalPulseShape::TEMPLATESAMPLES+1)/2)];
    for(int k=0; k<EcalPulseShape::TEMPLATESAMPLES*(EcalPulseShape::TEMPLATESAMPLES+1)/2; ++k) {
      covs_node[k] = getChildNode(cellnode,pulsecov_tag[k]); 
      GetNodeData(covs_node[k], covs[k]);  
    }

    for(int k=0; k<EcalPulseShape::TEMPLATESAMPLES*(EcalPulseShape::TEMPLATESAMPLES+1)/2; ++k) record[detid].covval[k] = covs[k];

    cellnode = cellnode->getNextSibling();

    while(cellnode && cellnode->getNodeType() != DOMNode::ELEMENT_NODE)
      cellnode = cellnode->getNextSibling();
    chan++;
  } 

  delete parser;
  cms::concurrency::xercesTerminate();
  std::cout << " nb of channels found in xml file " << chan << std::endl;
  return 0;
 }

int EcalPulseSymmCovariancesXMLTranslator::writeXML(const std::string& filename, 
					  const EcalCondHeader& header,
					  const EcalPulseSymmCovariances& record){
  std::fstream fs(filename.c_str(),ios::out);
  fs<< dumpXML(header,record);
  return 0;  
}


std::string EcalPulseSymmCovariancesXMLTranslator::dumpXML(const EcalCondHeader& header,const EcalPulseSymmCovariances& record){

  cms::concurrency::xercesInitialize();
  DOMImplementation*  impl =
    DOMImplementationRegistry::getDOMImplementation(fromNative("LS").c_str());

  DOMWriter* writer =static_cast<DOMImplementationLS*>(impl)->createDOMWriter( );
  writer->setFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true);

  DOMDocumentType* doctype = impl->createDocumentType(fromNative("XML").c_str(), 0, 0 );
  DOMDocument *    doc = 
    impl->createDocument( 0, fromNative(PulseSymmCovariances_tag).c_str(), doctype );

  doc->setEncoding(fromNative("UTF-8").c_str() );
  doc->setStandalone(true);
  doc->setVersion(fromNative("1.0").c_str() );
    
  DOMElement* root = doc->getDocumentElement();

  std::vector<std::string> pulsecov_tag(static_cast<size_t>(std::pow(EcalPulseShape::TEMPLATESAMPLES,2)));
  for(int k=0; k<std::pow(EcalPulseShape::TEMPLATESAMPLES,2); ++k) {
    int i = k/EcalPulseShape::TEMPLATESAMPLES;
    int j = k%EcalPulseShape::TEMPLATESAMPLES;
    int linK=-1;
    if(j >= i) linK = j + (EcalPulseShape::TEMPLATESAMPLES-1)*i;
    else linK = i + (EcalPulseShape::TEMPLATESAMPLES-1)*j;
    pulsecov_tag[linK] = Form("samplecov_%d_%d",i,j);
  }

  xuti::writeHeader(root,header);
  if (!record.barrelItems().size()) return std::string();
  for(int cellid = EBDetId::MIN_HASH;
      cellid < EBDetId::kSizeForDenseIndexing;
      ++cellid) {
    uint32_t rawid = EBDetId::unhashIndex(cellid);

    if (record.find(rawid) == record.end()) continue;

    DOMElement* cellnode=writeCell(root,rawid);

    for(int k=0; k<std::pow(EcalPulseShape::TEMPLATESAMPLES,2); ++k) { 
      WriteNodeWithValue(cellnode,pulsecov_tag[k],record[rawid].covval[k]);
    }
    
  }
  if (!record.endcapItems().size()) return std::string();
  for(int cellid = 0;
	cellid < EEDetId::kSizeForDenseIndexing;
	++cellid) {
    if(!EEDetId::validHashIndex(cellid)) continue;

    uint32_t rawid = EEDetId::unhashIndex(cellid); 

    if (record.find(rawid) == record.end()) continue;

    DOMElement* cellnode=writeCell(root,rawid);

    for(int k=0; k<std::pow(EcalPulseShape::TEMPLATESAMPLES,2); ++k) { 
      WriteNodeWithValue(cellnode,pulsecov_tag[k],record[rawid].covval[k]);
    }
  }

  std::string dump= toNative(writer->writeToString(*root)); 
  doc->release(); 
  return dump;
}
