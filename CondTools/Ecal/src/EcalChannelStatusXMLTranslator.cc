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


#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondTools/Ecal/interface/EcalChannelStatusXMLTranslator.h"

#include "CondTools/Ecal/interface/DOMHelperFunctions.h"
#include "CondTools/Ecal/interface/XMLTags.h"

using namespace XERCES_CPP_NAMESPACE;
using namespace xuti;
using namespace std;

int  EcalChannelStatusXMLTranslator::readXML(const std::string& filename, 
					     EcalCondHeader& header,
					     EcalChannelStatus& record){

  cms::concurrency::xercesInitialize();

  XercesDOMParser* parser = new XercesDOMParser;
  parser->setValidationScheme( XercesDOMParser::Val_Never );
  parser->setDoNamespaces( false );
  parser->setDoSchema( false );
  
  parser->parse(filename.c_str());

  DOMDocument* xmlDoc = parser->getDocument();
  if (!xmlDoc) {
    std::cout << "EcalChannelStatusXMLTranslator::Error parsing document" << std::endl;
    return -1;
  }

  DOMElement* elementRoot = xmlDoc->getDocumentElement();

  xuti::readHeader(elementRoot,header);

  DOMNode * cellnode = getChildNode(elementRoot,Cell_tag);

  while(cellnode)
    {
      uint16_t csc = 0;

      DetId detid = readCellId(dynamic_cast<DOMElement*>(cellnode));

      DOMNode* c_node = getChildNode(cellnode,ChannelStatusCode_tag);
      GetNodeData(c_node,csc);

      EcalChannelStatusCode ecalCSC = EcalChannelStatusCode(csc);
      record[detid] = ecalCSC;

      cellnode = cellnode->getNextSibling();

      while(cellnode && cellnode->getNodeType() != DOMNode::ELEMENT_NODE)
	cellnode = cellnode->getNextSibling();

 
    }  

  delete parser;
  cms::concurrency::xercesTerminate();
  return 0; 
}

int
EcalChannelStatusXMLTranslator::writeXML(const std::string& filename, 
					 const EcalCondHeader& header,
					 const EcalChannelStatus& record) {
  cms::concurrency::xercesInitialize();

  std::fstream fs(filename.c_str(),ios::out);
  fs<< dumpXML(header,record);

  cms::concurrency::xercesTerminate();

  return 0;
}


std::string
EcalChannelStatusXMLTranslator::dumpXML(const EcalCondHeader& header,
					const EcalChannelStatus& record) {

  unique_ptr<DOMImplementation> impl( DOMImplementationRegistry::getDOMImplementation( cms::xerces::uStr("LS").ptr()));
  
  DOMLSSerializer* writer = impl->createLSSerializer();
  if( writer->getDomConfig()->canSetParameter( XMLUni::fgDOMWRTFormatPrettyPrint, true ))
    writer->getDomConfig()->setParameter( XMLUni::fgDOMWRTFormatPrettyPrint, true );

  DOMDocumentType* doctype = impl->createDocumentType( cms::xerces::uStr("XML").ptr(), nullptr, nullptr );
  DOMDocument* doc = 
    impl->createDocument( nullptr, cms::xerces::uStr(ChannelStatus_tag.c_str()).ptr(), doctype );
    
  DOMElement* root = doc->getDocumentElement();
 
  xuti::writeHeader(root,header);

  for(int cellid = EBDetId::MIN_HASH;
      cellid < EBDetId::kSizeForDenseIndexing;
      ++cellid) {

    uint32_t rawid = EBDetId::unhashIndex(cellid);

    if(!record[rawid].getStatusCode()) continue;
    
    DOMElement* cellnode = writeCell(root,rawid);	  
    
    WriteNodeWithValue(cellnode,ChannelStatusCode_tag,record[rawid].getStatusCode());	  	  	  
  }

  for(int cellid = 0;
      cellid < EEDetId::kSizeForDenseIndexing;
      ++cellid) {
	  
    if(!EEDetId::validHashIndex(cellid)) continue;
	  
    uint32_t rawid = EEDetId::unhashIndex(cellid); 

    if(!record[rawid].getStatusCode()) continue;

    DOMElement* cellnode = writeCell(root,rawid);
    WriteNodeWithValue(cellnode,ChannelStatusCode_tag,record[rawid].getStatusCode());  
  }
    
  std::string dump = cms::xerces::toString(writer->writeToString( root )); 
  doc->release();
  doctype->release();
  writer->release();

  return dump;
}
