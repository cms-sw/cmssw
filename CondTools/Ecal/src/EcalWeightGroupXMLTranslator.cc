#include <iostream>
#include <sstream>
#include <xercesc/dom/DOMNode.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/PlatformUtils.hpp>
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

  XMLPlatformUtils::Initialize();

  XercesDOMParser* parser = new XercesDOMParser;
  parser->setValidationScheme( XercesDOMParser::Val_Never );
  parser->setDoNamespaces( false );
  parser->setDoSchema( false );
  
  parser->parse(filename.c_str());

  DOMDocument* xmlDoc = parser->getDocument();
  if (!xmlDoc) {
    cout << "EcalWeightGroupXMLTranslator::Error parsing document" << endl;
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
    DetId detid= readCellId(cellnode);
       
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
  XMLPlatformUtils::Terminate();
  return 0;
    
}





int EcalWeightGroupXMLTranslator::writeXML(const std::string& filename, 
					   const EcalCondHeader& header,
					   const EcalWeightXtalGroups& record){
    


  
  XMLPlatformUtils::Initialize();
  
  DOMImplementation*  impl =
    DOMImplementationRegistry::getDOMImplementation(fromNative("LS").c_str());
  
  DOMWriter* writer =static_cast<DOMImplementationLS*>(impl)->createDOMWriter( );
  writer->setFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true);
  
  DOMDocumentType* doctype = impl->createDocumentType( fromNative("XML").c_str(), 0, 0 );
  DOMDocument *    doc = 
    impl->createDocument( 0, fromNative(WeightGroups_tag).c_str(), doctype );
  
  
  doc->setEncoding(fromNative("UTF-8").c_str() );
  doc->setStandalone(true);
  doc->setVersion(fromNative("1.0").c_str() );
  
  
  DOMElement* root = doc->getDocumentElement();
 
  xuti::writeHeader(root,header);

  for (int cellid = EBDetId::MIN_HASH; 
       cellid < EBDetId::kSizeForDenseIndexing; 
       ++cellid){// loop on EB cells
    
    uint32_t rawid= EBDetId::unhashIndex(cellid);

    //if (!record[rawid]) continue; // cell absent from original record
    
    DOMElement* cellnode = doc->createElement(fromNative(Cell_tag).c_str());
    root->appendChild(cellnode);
     
    writeCellId(cellnode,rawid);

    WriteNodeWithValue(cellnode,WeightGroup_tag,record[rawid].id());
  

  } // loop on EB cells
  
  
  
  for (int cellid = 0; 
       cellid < EEDetId::kSizeForDenseIndexing; 
       ++cellid){// loop on EE cells
    
    if (!EEDetId::validHashIndex(cellid)) continue;

    uint32_t rawid= EEDetId::unhashIndex(cellid);
    //xif (!record[rawid]) continue; // cell absent from original record

    DOMElement* cellnode = doc->createElement(fromNative(Cell_tag).c_str());
    root->appendChild(cellnode);
        
    writeCellId(cellnode,rawid);
    WriteNodeWithValue(cellnode,WeightGroup_tag,record[rawid].id());
    
  } // loop on EE cells
  
  
  
  LocalFileFormatTarget file(filename.c_str());
  
  writer->writeNode(&file, *root);
  
  doc->release();
  //   XMLPlatformUtils::Terminate();
  
  return 0;
}


