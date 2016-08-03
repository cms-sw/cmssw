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


#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondTools/Ecal/interface/EcalADCToGeVXMLTranslator.h"
#include "CondTools/Ecal/interface/DOMHelperFunctions.h"

using namespace XERCES_CPP_NAMESPACE;
using namespace xuti;
using namespace std;

 

int  EcalADCToGeVXMLTranslator::readXML(const std::string& filename, 
					EcalCondHeader& header,
					EcalADCToGeVConstant& record){

 
  cms::concurrency::xercesInitialize();

  XercesDOMParser* parser = new XercesDOMParser;
  parser->setValidationScheme( XercesDOMParser::Val_Never );
  parser->setDoNamespaces( false );
  parser->setDoSchema( false );
  
  parser->parse(filename.c_str());

  DOMDocument* xmlDoc = parser->getDocument();
  if (!xmlDoc) {
    std::cout << "EcalADCToGeVXMLTranslator::Error parsing document" << std::endl;
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
  cms::concurrency::xercesTerminate();
  return 0;

}





int EcalADCToGeVXMLTranslator::writeXML(const std::string& filename, 
					const EcalCondHeader& header,
					const EcalADCToGeVConstant& record){
  std::fstream fs(filename.c_str(),ios::out);
  fs<< dumpXML(header,record);
  return 0;  
 
}

std::string EcalADCToGeVXMLTranslator::dumpXML(const EcalCondHeader& header,
					  const EcalADCToGeVConstant& record){

  cms::concurrency::xercesInitialize();
  
  unique_ptr<DOMImplementation> impl( DOMImplementationRegistry::getDOMImplementation(fromNative("LS").c_str()));
  
  DOMLSSerializer* writer = impl->createLSSerializer();
  if( writer->getDomConfig()->canSetParameter( XMLUni::fgDOMWRTFormatPrettyPrint, true ))
    writer->getDomConfig()->setParameter( XMLUni::fgDOMWRTFormatPrettyPrint, true );
  
  DOMDocumentType* doctype = impl->createDocumentType(fromNative("XML").c_str(), 0, 0 );
  DOMDocument *    doc = 
    impl->createDocument( 0, fromNative(ADCToGeVConstant_tag).c_str(), doctype );
    
  DOMElement* root = doc->getDocumentElement();
 
  xuti::writeHeader(root,header);
   
  xuti::WriteNodeWithValue(root,Barrel_tag,record.getEBValue());
  xuti::WriteNodeWithValue(root,Endcap_tag,record.getEEValue());

  std::string dump = toNative(writer->writeToString( root )); 
  doc->release();
  doctype->release();
  writer->release();
  //   cms::concurrency::xercesTerminate();

  return dump;
}

