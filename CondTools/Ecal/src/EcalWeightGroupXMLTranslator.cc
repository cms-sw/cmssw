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

#include "CondTools/Ecal/interface/EcalWeightGroupXMLTranslator.h"
#include "CondFormats/EcalObjects/interface/EcalXtalGroupId.h"
#include "CondTools/Ecal/interface/DOMHelperFunctions.h"

using namespace XERCES_CPP_NAMESPACE;
using namespace xuti;
using namespace std;

int  EcalWeightGroupXMLTranslator::readXML(const std::string& filename, 
					   EcalCondHeader& header,
					   EcalWeightXtalGroups& record){

  cms::concurrency::xercesInitialize();

  XercesDOMParser* parser = new XercesDOMParser;
  parser->setValidationScheme( XercesDOMParser::Val_Never );
  parser->setDoNamespaces( false );
  parser->setDoSchema( false );
  
  parser->parse(filename.c_str());

  DOMDocument* xmlDoc = parser->getDocument();
  if (!xmlDoc) {
    std::cout << "EcalWeightGroupXMLTranslator::Error parsing document" << std::endl;
    return -1;
  }

  DOMElement* elementRoot = xmlDoc->getDocumentElement();

  xuti::readHeader(elementRoot,header);
  // get the first cell node
  DOMNode * cellnode=getChildNode(elementRoot,Cell_tag);
  
  // loop on cell nodes
  while  (cellnode){

    unsigned int  group=0;

    // read id
    DetId detid= readCellId(dynamic_cast<DOMElement*> (cellnode));
       
    // read constant
    DOMNode * c_node = getChildNode(cellnode,WeightGroup_tag);
    GetNodeData(c_node,group);

    // fill record
    record[detid]=EcalXtalGroupId(group);
   
    // get next cell
    cellnode= cellnode->getNextSibling();
    
    while (cellnode&& cellnode->getNodeType( ) != DOMNode::ELEMENT_NODE)      
      cellnode= cellnode->getNextSibling();
    
  }

  delete parser;
  cms::concurrency::xercesTerminate();
  return 0;
    
}

int EcalWeightGroupXMLTranslator::writeXML(const std::string& filename, 
					   const EcalCondHeader& header,
					   const EcalWeightXtalGroups& record){
  cms::concurrency::xercesInitialize();

  std::fstream fs(filename.c_str(),ios::out);
  fs<< dumpXML(header,record);

  cms::concurrency::xercesTerminate();

  return 0;
}

std::string 
EcalWeightGroupXMLTranslator::dumpXML(const EcalCondHeader& header,
				      const EcalWeightXtalGroups& record){

  unique_ptr<DOMImplementation> impl( DOMImplementationRegistry::getDOMImplementation(cms::xerces::uStr("LS").ptr()));
  
  DOMLSSerializer* writer = impl->createLSSerializer();
  if( writer->getDomConfig()->canSetParameter( XMLUni::fgDOMWRTFormatPrettyPrint, true ))
    writer->getDomConfig()->setParameter( XMLUni::fgDOMWRTFormatPrettyPrint, true );
  
  DOMDocumentType* doctype = impl->createDocumentType( cms::xerces::uStr("XML").ptr(), nullptr, nullptr );
  DOMDocument* doc = impl->createDocument( nullptr, cms::xerces::uStr(WeightGroups_tag.c_str()).ptr(), doctype );
  DOMElement* root = doc->getDocumentElement();
 
  xuti::writeHeader(root,header);

  for (int cellid = EBDetId::MIN_HASH; 
       cellid < EBDetId::kSizeForDenseIndexing; 
       ++cellid){// loop on EB cells
    
    uint32_t rawid= EBDetId::unhashIndex(cellid);

    //if (!record[rawid]) continue; // cell absent from original record
    
    DOMElement* cellnode=writeCell(root,rawid);

    WriteNodeWithValue(cellnode,WeightGroup_tag,record[rawid].id());
  

  } // loop on EB cells
  
  
  
  for (int cellid = 0; 
       cellid < EEDetId::kSizeForDenseIndexing; 
       ++cellid){// loop on EE cells
    
    if (!EEDetId::validHashIndex(cellid)) continue;

    uint32_t rawid= EEDetId::unhashIndex(cellid);
    //xif (!record[rawid]) continue; // cell absent from original record

        
    DOMElement* cellnode=writeCell(root,rawid);
    WriteNodeWithValue(cellnode,WeightGroup_tag,record[rawid].id());
    
  } // loop on EE cells
  
  std::string dump = cms::xerces::toString( writer->writeToString( root ));
  doc->release();
  doctype->release();
  writer->release();
  
  return dump;
}
