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


#include "CondFormats/ESObjects/interface/ESGain.h"
#include "CondTools/Ecal/interface/ESGainXMLTranslator.h"
#include "CondTools/Ecal/interface/DOMHelperFunctions.h"

using namespace XERCES_CPP_NAMESPACE;
using namespace xuti;
using namespace std;

int ESGainXMLTranslator::writeXML(const std::string& filename, 
					const EcalCondHeader& header,
					const ESGain& record){
  std::fstream fs(filename.c_str(),ios::out);
  fs<< dumpXML(header,record);
  return 0;  
 
}

std::string ESGainXMLTranslator::dumpXML(const EcalCondHeader& header,
					  const ESGain& record){

  XMLPlatformUtils::Initialize();
  
  DOMImplementation*  impl =
    DOMImplementationRegistry::getDOMImplementation(fromNative("LS").c_str());
  
  DOMWriter* writer =static_cast<DOMImplementationLS*>(impl)->createDOMWriter( );
  writer->setFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true);
  
  DOMDocumentType* doctype = impl->createDocumentType(fromNative("XML").c_str(), 0, 0 );
  DOMDocument *    doc = 
    impl->createDocument( 0, fromNative(ADCToGeVConstant_tag).c_str(), doctype );


  doc->setEncoding(fromNative("UTF-8").c_str() );
  doc->setStandalone(true);
  doc->setVersion(fromNative("1.0").c_str() );
    
  DOMElement* root = doc->getDocumentElement();
 
  xuti::writeHeader(root,header);
   
  xuti::WriteNodeWithValue(root,ESGain_tag,record.getESGain());

  std::string dump= toNative(writer->writeToString(*root)); 
  doc->release();

  //   XMLPlatformUtils::Terminate();

  return dump;
}

