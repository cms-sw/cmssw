#include <iostream>
#include <sstream>
#include <fstream>
#include <xercesc/dom/DOMNode.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include "FWCore/Concurrency/interface/Xerces.h"
#include <xercesc/util/XMLString.hpp>
#include <xercesc/sax/SAXException.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>


#include "CondFormats/EcalObjects/interface/EcalPulseShapes.h"
#include "CondTools/Ecal/interface/EcalPulseShapesXMLTranslator.h"
#include "CondTools/Ecal/interface/DOMHelperFunctions.h"
#include "CondTools/Ecal/interface/XMLTags.h"


using namespace XERCES_CPP_NAMESPACE;
using namespace xuti;
using namespace std;

int  EcalPulseShapesXMLTranslator::readXML(const std::string& filename, 
					  EcalCondHeader& header,
					  EcalPulseShapes& record){

  cms::concurrency::xercesInitialize();

  XercesDOMParser* parser = new XercesDOMParser;
  parser->setValidationScheme( XercesDOMParser::Val_Never );
  parser->setDoNamespaces( false );
  parser->setDoSchema( false );
  
  parser->parse(filename.c_str());

  DOMDocument* xmlDoc = parser->getDocument();
  if (!xmlDoc) {
    std::cout << "EcalPulseShapesXMLTranslator::Error parsing document" << std::endl;
    return -1;
  }

  DOMElement* elementRoot = xmlDoc->getDocumentElement();

  xuti::readHeader(elementRoot,header);

  DOMNode * cellnode = getChildNode(elementRoot,Cell_tag);

  int chan = 0;
  while(cellnode) {
    //    std::cout << " Channel " << chan << std::endl;
    float samples[EcalPulseShape::TEMPLATESAMPLES];

    DetId detid = readCellId(dynamic_cast<DOMElement*>(cellnode));
    //    std::cout << " readCell Id Channel " << chan << " tag " << mean12_tag << std::endl;

    DOMNode* sample0_node = getChildNode(cellnode,sample0_tag);
    GetNodeData(sample0_node,samples[0]);
    //    std::cout << " tag " << sample0_tag << " sample0 " << sample0 << std::endl;

    DOMNode* sample1_node = getChildNode(cellnode,sample1_tag);
    GetNodeData(sample1_node,samples[1]);
    //    std::cout << " tag " << sample1_tag << " sample1 " << sample1 << std::endl;

    DOMNode* sample2_node = getChildNode(cellnode,sample2_tag);
    GetNodeData(sample2_node,samples[2]);
    //    std::cout << " tag " << sample2_tag << " sample2 " << sample2 << std::endl;

    DOMNode* sample3_node = getChildNode(cellnode,sample3_tag);
    GetNodeData(sample3_node,samples[3]);
    //    std::cout << " tag " << sample3_tag << " sample3 " << sample3 << std::endl;

    DOMNode* sample4_node = getChildNode(cellnode,sample4_tag);
    GetNodeData(sample4_node,samples[4]);
    //    std::cout << " tag " << sample4_tag << " sample4 " << sample4 << std::endl;

    DOMNode* sample5_node = getChildNode(cellnode,sample5_tag);
    GetNodeData(sample5_node,samples[5]);
    //    std::cout << " tag " << sample5_tag << " sample5 " << sample5 << std::endl;

    DOMNode* sample6_node = getChildNode(cellnode,sample6_tag);
    GetNodeData(sample6_node,samples[6]);
    //    std::cout << " tag " << sample6_tag << " sample6 " << sample6 << std::endl;

    DOMNode* sample7_node = getChildNode(cellnode,sample7_tag);
    GetNodeData(sample7_node,samples[7]);
    //    std::cout << " tag " << sample7_tag << " sample7 " << sample7 << std::endl;

    DOMNode* sample8_node = getChildNode(cellnode,sample8_tag);
    GetNodeData(sample8_node,samples[8]);
    //    std::cout << " tag " << sample8_tag << " sample8 " << sample8 << std::endl;

    DOMNode* sample9_node = getChildNode(cellnode,sample9_tag);
    GetNodeData(sample9_node,samples[9]);
    //    std::cout << " tag " << sample9_tag << " sample9 " << sample9 << std::endl;

    DOMNode* sample10_node = getChildNode(cellnode,sample10_tag);
    GetNodeData(sample10_node,samples[10]);
    //    std::cout << " tag " << sample10_tag << " sample10 " << sample10 << std::endl;

    DOMNode* sample11_node = getChildNode(cellnode,sample11_tag);
    GetNodeData(sample11_node,samples[11]);
    //    std::cout << " tag " << sample11_tag << " sample11 " << sample11 << std::endl;

    for(int s = 0; s<EcalPulseShape::TEMPLATESAMPLES; ++s) record[detid].pdfval[s] = samples[s];

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

int EcalPulseShapesXMLTranslator::writeXML(const std::string& filename, 
					  const EcalCondHeader& header,
					  const EcalPulseShapes& record){
  std::fstream fs(filename.c_str(),ios::out);
  fs<< dumpXML(header,record);
  return 0;  
}


std::string EcalPulseShapesXMLTranslator::dumpXML(const EcalCondHeader& header,const EcalPulseShapes& record){

  cms::concurrency::xercesInitialize();
  DOMImplementation*  impl =
    DOMImplementationRegistry::getDOMImplementation(fromNative("LS").c_str());

  DOMWriter* writer =static_cast<DOMImplementationLS*>(impl)->createDOMWriter( );
  writer->setFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true);

  DOMDocumentType* doctype = impl->createDocumentType(fromNative("XML").c_str(), 0, 0 );
  DOMDocument *    doc = 
    impl->createDocument( 0, fromNative(PulseShapes_tag).c_str(), doctype );

  doc->setEncoding(fromNative("UTF-8").c_str() );
  doc->setStandalone(true);
  doc->setVersion(fromNative("1.0").c_str() );
    
  DOMElement* root = doc->getDocumentElement();

  xuti::writeHeader(root,header);
  if (!record.barrelItems().size()) return std::string();
  for(int cellid = EBDetId::MIN_HASH;
      cellid < EBDetId::kSizeForDenseIndexing;
      ++cellid) {
    uint32_t rawid = EBDetId::unhashIndex(cellid);

    if (record.find(rawid) == record.end()) continue;
    if(!record[rawid].pdfval[5]) continue;

    DOMElement* cellnode=writeCell(root,rawid);

    WriteNodeWithValue(cellnode,sample0_tag,record[rawid].pdfval[0]);
    WriteNodeWithValue(cellnode,sample1_tag,record[rawid].pdfval[1]);
    WriteNodeWithValue(cellnode,sample2_tag,record[rawid].pdfval[2]);
    WriteNodeWithValue(cellnode,sample3_tag,record[rawid].pdfval[3]);
    WriteNodeWithValue(cellnode,sample4_tag,record[rawid].pdfval[4]);
    WriteNodeWithValue(cellnode,sample5_tag,record[rawid].pdfval[5]);
    WriteNodeWithValue(cellnode,sample6_tag,record[rawid].pdfval[6]);
    WriteNodeWithValue(cellnode,sample7_tag,record[rawid].pdfval[7]);
    WriteNodeWithValue(cellnode,sample8_tag,record[rawid].pdfval[8]);
    WriteNodeWithValue(cellnode,sample9_tag,record[rawid].pdfval[9]);
    WriteNodeWithValue(cellnode,sample10_tag,record[rawid].pdfval[10]);
    WriteNodeWithValue(cellnode,sample11_tag,record[rawid].pdfval[11]);

  }

  if (!record.endcapItems().size()) return std::string();
  for(int cellid = 0;
	cellid < EEDetId::kSizeForDenseIndexing;
	++cellid) {
    if(!EEDetId::validHashIndex(cellid)) continue;

    uint32_t rawid = EEDetId::unhashIndex(cellid); 

    if (record.find(rawid) == record.end()) continue;
    if(!record[rawid].pdfval[5]) continue;

    DOMElement* cellnode=writeCell(root,rawid);

    WriteNodeWithValue(cellnode,sample0_tag,record[rawid].pdfval[0]);
    WriteNodeWithValue(cellnode,sample1_tag,record[rawid].pdfval[1]);
    WriteNodeWithValue(cellnode,sample2_tag,record[rawid].pdfval[2]);
    WriteNodeWithValue(cellnode,sample3_tag,record[rawid].pdfval[3]);
    WriteNodeWithValue(cellnode,sample4_tag,record[rawid].pdfval[4]);
    WriteNodeWithValue(cellnode,sample5_tag,record[rawid].pdfval[5]);
    WriteNodeWithValue(cellnode,sample6_tag,record[rawid].pdfval[6]);
    WriteNodeWithValue(cellnode,sample7_tag,record[rawid].pdfval[7]);
    WriteNodeWithValue(cellnode,sample8_tag,record[rawid].pdfval[8]);
    WriteNodeWithValue(cellnode,sample9_tag,record[rawid].pdfval[9]);
    WriteNodeWithValue(cellnode,sample10_tag,record[rawid].pdfval[10]);
    WriteNodeWithValue(cellnode,sample11_tag,record[rawid].pdfval[11]);

  }

  std::string dump= toNative(writer->writeToString(*root)); 
  doc->release(); 
  return dump;
}
