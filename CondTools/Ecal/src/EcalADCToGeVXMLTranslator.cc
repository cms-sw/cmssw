#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondTools/Ecal/interface/EcalADCToGeVXMLTranslator.h"
#include "CondTools/Ecal/interface/DOMHelperFunctions.h"
#include "FWCore/Concurrency/interface/Xerces.h"
#include "Utilities/Xerces/interface/XercesStrUtils.h"
#include <xercesc/dom/DOMNode.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/sax/SAXException.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <iostream>
#include <sstream>
#include <fstream>

using namespace XERCES_CPP_NAMESPACE;
using namespace xuti;
using namespace std;


int  EcalADCToGeVXMLTranslator::readXML(const std::string& filename, 
					EcalCondHeader& header,
					EcalADCToGeVConstant& record) {
 
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
					const EcalADCToGeVConstant& record) {

  cms::concurrency::xercesInitialize();

  std::fstream fs(filename.c_str(),ios::out);
  fs<< dumpXML(header,record);
  
  cms::concurrency::xercesTerminate();

  return 0;
}

std::string
EcalADCToGeVXMLTranslator::dumpXML(const EcalCondHeader& header,
				   const EcalADCToGeVConstant& record) {

  unique_ptr<DOMImplementation> impl( DOMImplementationRegistry::getDOMImplementation( cms::xerces::uStr("LS").ptr()));
  
  DOMLSSerializer* writer = impl->createLSSerializer();
  if( writer->getDomConfig()->canSetParameter( XMLUni::fgDOMWRTFormatPrettyPrint, true ))
    writer->getDomConfig()->setParameter( XMLUni::fgDOMWRTFormatPrettyPrint, true );
  
  DOMDocumentType* doctype = impl->createDocumentType( cms::xerces::uStr("XML").ptr(), nullptr, nullptr );
  DOMDocument* doc = 
    impl->createDocument( nullptr, cms::xerces::uStr(ADCToGeVConstant_tag.c_str()).ptr(), doctype );
    
  DOMElement* root = doc->getDocumentElement();
 
  xuti::writeHeader(root,header);
   
  xuti::WriteNodeWithValue(root,Barrel_tag,record.getEBValue());
  xuti::WriteNodeWithValue(root,Endcap_tag,record.getEEValue());

  std::string dump = cms::xerces::toString(writer->writeToString( root )); 
  doc->release();
  doctype->release();
  writer->release();

  return dump;
}

