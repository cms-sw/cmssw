#include <iostream>
#include <sstream>
#include <fstream>
#include <xercesc/dom/DOMNode.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include "FWCore/Concurrency/interface/Xerces.h"
#include "Utilities/Xerces/interface/XercesStrUtils.h"
#include <xercesc/util/XMLString.hpp>
#include <xercesc/sax/SAXException.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>


#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondTools/Ecal/interface/EcalPedestalsXMLTranslator.h"
#include "CondTools/Ecal/interface/DOMHelperFunctions.h"
#include "CondTools/Ecal/interface/XMLTags.h"


using namespace XERCES_CPP_NAMESPACE;
using namespace xuti;
using namespace std;

int  EcalPedestalsXMLTranslator::readXML(const std::string& filename, 
					  EcalCondHeader& header,
					  EcalPedestals& record){

  cms::concurrency::xercesInitialize();

  XercesDOMParser* parser = new XercesDOMParser;
  parser->setValidationScheme( XercesDOMParser::Val_Never );
  parser->setDoNamespaces( false );
  parser->setDoSchema( false );
  
  parser->parse(filename.c_str());

  DOMDocument* xmlDoc = parser->getDocument();
  if (!xmlDoc) {
    std::cout << "EcalPedestalsXMLTranslator::Error parsing document" << std::endl;
    return -1;
  }

  DOMElement* elementRoot = xmlDoc->getDocumentElement();

  xuti::readHeader(elementRoot,header);

  DOMNode * cellnode = getChildNode(elementRoot,Cell_tag);

  int chan = 0;
  while(cellnode) {
    //    std::cout << " Channel " << chan << std::endl;
    float mean12 = 0;
    float mean6 = 0;
    float mean1 = 0;
    float rms12 = 0;
    float rms6 = 0;
    float rms1 = 0;
    DetId detid = readCellId(dynamic_cast<DOMElement*>(cellnode));
    //    std::cout << " readCell Id Channel " << chan << " tag " << mean12_tag << std::endl;

    DOMNode* mean12_node = getChildNode(cellnode,mean12_tag);
    GetNodeData(mean12_node,mean12);
    //    std::cout << " tag " << mean12_tag << " mean12 " << mean12 << std::endl;

    DOMNode* mean6_node = getChildNode(cellnode,mean6_tag);
    GetNodeData(mean6_node,mean6);
    //    std::cout << " tag " << mean6_tag << " mean6 " << mean6 << std::endl;

    DOMNode* mean1_node = getChildNode(cellnode,mean1_tag);
    GetNodeData(mean1_node,mean1);
    //    std::cout << " tag " << mean1_tag << " mean1 " << mean1 << std::endl;

    DOMNode* rms12_node = getChildNode(cellnode,rms12_tag);
    GetNodeData(rms12_node,rms12);
    //    std::cout << " tag 12 " << rms12_tag << " rms 12 " << rms12 << std::endl;

    DOMNode* rms6_node = getChildNode(cellnode,rms6_tag);
    GetNodeData(rms6_node,rms6);
    //    std::cout << " tag 6 " << rms6_tag << " rms 6 " << rms6 << std::endl;

    DOMNode* rms1_node = getChildNode(cellnode,rms1_tag);
    //    std::cout << " tag 1 " << rms1_tag << std::endl;

    GetNodeData(rms1_node,rms1);
    //    std::cout << " Channel " << chan << " mean12 " << mean12 << " rms 1 " << rms1 << std::endl;
    record[detid].mean_x12 = mean12;
    record[detid].mean_x6 = mean6;
    record[detid].mean_x1 = mean1;
    record[detid].rms_x12 = rms12;
    record[detid].rms_x6 = rms6;
    record[detid].rms_x1 = rms1;

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

int EcalPedestalsXMLTranslator::writeXML(const std::string& filename, 
					  const EcalCondHeader& header,
					  const EcalPedestals& record){
  cms::concurrency::xercesInitialize();

  std::fstream fs(filename.c_str(),ios::out);
  fs<< dumpXML(header,record);

  cms::concurrency::xercesTerminate();

  return 0;  
}


std::string EcalPedestalsXMLTranslator::dumpXML(const EcalCondHeader& header,const EcalPedestals& record){

  unique_ptr<DOMImplementation> impl( DOMImplementationRegistry::getDOMImplementation(cms::xerces::uStr("LS").ptr()));
  
  DOMLSSerializer* writer = impl->createLSSerializer();
  if( writer->getDomConfig()->canSetParameter( XMLUni::fgDOMWRTFormatPrettyPrint, true ))
    writer->getDomConfig()->setParameter( XMLUni::fgDOMWRTFormatPrettyPrint, true );

  DOMDocumentType* doctype = impl->createDocumentType(cms::xerces::uStr("XML").ptr(), nullptr, nullptr );
  DOMDocument *    doc = 
    impl->createDocument( nullptr, cms::xerces::uStr(Pedestals_tag.c_str()).ptr(), doctype );
    
  DOMElement* root = doc->getDocumentElement();

  xuti::writeHeader(root,header);
  if (record.barrelItems().empty()) return std::string();
  for(int cellid = EBDetId::MIN_HASH;
      cellid < EBDetId::kSizeForDenseIndexing;
      ++cellid) {
    uint32_t rawid = EBDetId::unhashIndex(cellid);

    if (record.find(rawid) == record.end()) continue;
    if(!record[rawid].mean_x12 && !record[rawid].rms_x12) continue;

    DOMElement* cellnode=writeCell(root,rawid);

    WriteNodeWithValue(cellnode,mean12_tag,record[rawid].mean_x12);
    WriteNodeWithValue(cellnode,mean6_tag,record[rawid].mean_x6);
    WriteNodeWithValue(cellnode,mean1_tag,record[rawid].mean_x1);
    WriteNodeWithValue(cellnode,rms12_tag,record[rawid].rms_x12);
    WriteNodeWithValue(cellnode,rms6_tag,record[rawid].rms_x6);
    WriteNodeWithValue(cellnode,rms1_tag,record[rawid].rms_x1);
  }

  if (record.endcapItems().empty()) return std::string();
  for(int cellid = 0;
	cellid < EEDetId::kSizeForDenseIndexing;
	++cellid) {
    if(!EEDetId::validHashIndex(cellid)) continue;

    uint32_t rawid = EEDetId::unhashIndex(cellid); 

    if (record.find(rawid) == record.end()) continue;
    if(!record[rawid].mean_x12 && !record[rawid].rms_x12) continue;

    DOMElement* cellnode=writeCell(root,rawid);

    WriteNodeWithValue(cellnode,mean12_tag,record[rawid].mean_x12);
    WriteNodeWithValue(cellnode,mean6_tag,record[rawid].mean_x6);
    WriteNodeWithValue(cellnode,mean1_tag,record[rawid].mean_x1);
    WriteNodeWithValue(cellnode,rms12_tag,record[rawid].rms_x12);
    WriteNodeWithValue(cellnode,rms6_tag,record[rawid].rms_x6);
    WriteNodeWithValue(cellnode,rms1_tag,record[rawid].rms_x1);
  }

  std::string dump = cms::xerces::toString(writer->writeToString( root )); 
  doc->release();
  doctype->release();
  writer->release();

  return dump;
}
