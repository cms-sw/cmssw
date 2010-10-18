#include <iostream>
#include <sstream>
#include <fstream>
#include <xercesc/dom/DOMNode.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/sax/SAXException.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>


#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondTools/Ecal/interface/EcalGainRatiosXMLTranslator.h"
#include "CondTools/Ecal/interface/DOMHelperFunctions.h"
#include "CondTools/Ecal/interface/XMLTags.h"


using namespace XERCES_CPP_NAMESPACE;
using namespace xuti;
using namespace std;

 




int  EcalGainRatiosXMLTranslator::readXML(const std::string& filename, 
					  EcalCondHeader& header,
					  EcalGainRatios& record){

 

  XMLPlatformUtils::Initialize();

  XercesDOMParser* parser = new XercesDOMParser;
  parser->setValidationScheme( XercesDOMParser::Val_Never );
  parser->setDoNamespaces( false );
  parser->setDoSchema( false );
  
  parser->parse(filename.c_str());

  DOMDocument* xmlDoc = parser->getDocument();
  if (!xmlDoc) {
    std::cout << "EcalGainRatiosXMLTranslator::Error parsing document" << std::endl;
    return -1;
  }

  DOMElement* elementRoot = xmlDoc->getDocumentElement();

  xuti::readHeader(elementRoot,header);

  DOMNode * cellnode = getChildNode(elementRoot,Cell_tag);

  while(cellnode)
    {
      float g12_6 = 0;
      float g6_1 = 0;
      DetId detid = readCellId(dynamic_cast<DOMElement*>(cellnode));

      DOMNode* g12_6_node = getChildNode(cellnode,Gain12Over6_tag);
      GetNodeData(g12_6_node,g12_6);

      DOMNode* g6_1_node = getChildNode(cellnode,Gain6Over1_tag);
      GetNodeData(g6_1_node,g6_1);

      record[detid].setGain12Over6(g12_6);
      record[detid].setGain6Over1(g6_1);

      cellnode = cellnode->getNextSibling();

      while(cellnode && cellnode->getNodeType() != DOMNode::ELEMENT_NODE)
	cellnode = cellnode->getNextSibling();

    } 

  delete parser;
  XMLPlatformUtils::Terminate();
  return 0;
  
  
}
  




int EcalGainRatiosXMLTranslator::writeXML(const std::string& filename, 
					  const EcalCondHeader& header,
					  const EcalGainRatios& record){
  std::fstream fs(filename.c_str(),ios::out);
  fs<< dumpXML(header,record);
  return 0;  
}


std::string EcalGainRatiosXMLTranslator::dumpXML(const EcalCondHeader& header,const EcalGainRatios& record){

    XMLPlatformUtils::Initialize();

    DOMImplementation*  impl =
      DOMImplementationRegistry::getDOMImplementation(fromNative("LS").c_str());

    DOMWriter* writer =static_cast<DOMImplementationLS*>(impl)->createDOMWriter( );
    writer->setFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true);

    DOMDocumentType* doctype = impl->createDocumentType(fromNative("XML").c_str(), 0, 0 );
    DOMDocument *    doc = 
         impl->createDocument( 0, fromNative(GainRatios_tag).c_str(), doctype );


    doc->setEncoding(fromNative("UTF-8").c_str() );
    doc->setStandalone(true);
    doc->setVersion(fromNative("1.0").c_str() );

    
    DOMElement* root = doc->getDocumentElement();
 


    xuti::writeHeader(root,header);
    if (!record.barrelItems().size()) return std::string();
    for(int cellid = EBDetId::MIN_HASH;
	cellid < EBDetId::kSizeForDenseIndexing;
	++cellid)
      {
	 
	uint32_t rawid = EBDetId::unhashIndex(cellid);

	if (record.find(rawid) == record.end()) continue;
       	if(!record[rawid].gain12Over6() && !record[rawid].gain6Over1()) continue;
	  
	DOMElement* cellnode=writeCell(root,rawid);

	WriteNodeWithValue(cellnode,Gain12Over6_tag,record[rawid].gain12Over6());
	WriteNodeWithValue(cellnode,Gain6Over1_tag,record[rawid].gain6Over1());
      }

    if (!record.endcapItems().size()) return std::string();
    for(int cellid = 0;
	cellid < EEDetId::kSizeForDenseIndexing;
	++cellid)
      {
	if(!EEDetId::validHashIndex(cellid)) continue;

	uint32_t rawid = EEDetId::unhashIndex(cellid); 

	if (record.find(rawid) == record.end()) continue;
	if(!record[rawid].gain12Over6() && !record[rawid].gain6Over1()) continue;

	DOMElement* cellnode=writeCell(root,rawid);

	WriteNodeWithValue(cellnode,Gain12Over6_tag,record[rawid].gain12Over6());
	WriteNodeWithValue(cellnode,Gain6Over1_tag,record[rawid].gain6Over1());
	  
	  
      }

 
    std::string dump= toNative(writer->writeToString(*root)); 
    doc->release(); 
    return dump;
}
