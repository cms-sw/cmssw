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


#include "CondFormats/EcalObjects/interface/EcalTimeOffsetConstant.h"
#include "CondTools/Ecal/interface/EcalTimeOffsetXMLTranslator.h"
#include "CondTools/Ecal/interface/DOMHelperFunctions.h"

using namespace XERCES_CPP_NAMESPACE;
using namespace xuti;
using namespace std;

 

int  EcalTimeOffsetXMLTranslator::readXML(const std::string& filename, 
					EcalCondHeader& header,
					EcalTimeOffsetConstant& record){

 
  XMLPlatformUtils::Initialize();

  XercesDOMParser* parser = new XercesDOMParser;
  parser->setValidationScheme( XercesDOMParser::Val_Never );
  parser->setDoNamespaces( false );
  parser->setDoSchema( false );
  
  parser->parse(filename.c_str());

  DOMDocument* xmlDoc = parser->getDocument();
  if (!xmlDoc) {
    std::cout << "EcalTimeOffsetXMLTranslator::Error parsing document" << std::endl;
    return -1;
  }
  
  // Get the top-level element
  DOMElement* elementRoot = xmlDoc->getDocumentElement();

  xuti::readHeader(elementRoot, header);

  
  DOMNode * barrelnode = xuti::getChildNode(elementRoot,Barrel_tag);
  DOMNode * endcapnode = xuti::getChildNode(elementRoot,Endcap_tag);
  
  double barrelvalue=0;
  double endcapvalue=0;

  xuti::GetNodeData(barrelnode,barrelvalue);
  xuti::GetNodeData(endcapnode,endcapvalue);

  record.setEBValue(barrelvalue); 
  record.setEEValue(endcapvalue);


  delete parser;
  XMLPlatformUtils::Terminate();
  return 0;

}





int EcalTimeOffsetXMLTranslator::writeXML(const std::string& filename, 
					const EcalCondHeader& header,
					const EcalTimeOffsetConstant& record){
  std::fstream fs(filename.c_str(),ios::out);
  fs<< dumpXML(header,record);
  return 0;  
 
}

std::string EcalTimeOffsetXMLTranslator::dumpXML(const EcalCondHeader& header,
					  const EcalTimeOffsetConstant& record){

  XMLPlatformUtils::Initialize();
  
  DOMImplementation*  impl =
    DOMImplementationRegistry::getDOMImplementation(fromNative("LS").c_str());
  
  DOMWriter* writer =static_cast<DOMImplementationLS*>(impl)->createDOMWriter( );
  writer->setFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true);
  
  DOMDocumentType* doctype = impl->createDocumentType(fromNative("XML").c_str(), 0, 0 );
  DOMDocument *    doc = 
    impl->createDocument( 0, fromNative(TimeOffsetConstant_tag).c_str(), doctype );


  doc->setEncoding(fromNative("UTF-8").c_str() );
  doc->setStandalone(true);
  doc->setVersion(fromNative("1.0").c_str() );
    
  DOMElement* root = doc->getDocumentElement();
 
  xuti::writeHeader(root,header);
   
  xuti::WriteNodeWithValue(root,Barrel_tag,record.getEBValue());
  xuti::WriteNodeWithValue(root,Endcap_tag,record.getEEValue());

  std::string dump= toNative(writer->writeToString(*root)); 
  doc->release();

  //   XMLPlatformUtils::Terminate();

  return dump;
}

