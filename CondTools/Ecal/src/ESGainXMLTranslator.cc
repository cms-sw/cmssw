#include "CondFormats/ESObjects/interface/ESGain.h"
#include "CondTools/Ecal/interface/ESGainXMLTranslator.h"
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

int
ESGainXMLTranslator::writeXML(const std::string& filename, 
			      const EcalCondHeader& header,
			      const ESGain& record) {
  
  cms::concurrency::xercesInitialize();

  std::fstream fs(filename.c_str(),ios::out);
  fs<< dumpXML(header,record);

  cms::concurrency::xercesTerminate();
  
  return 0;
}

std::string
ESGainXMLTranslator::dumpXML(const EcalCondHeader& header,
			     const ESGain& record) {
  
  unique_ptr<DOMImplementation> impl( DOMImplementationRegistry::getDOMImplementation( cms::xerces::uStr("LS").ptr()));
  
  DOMLSSerializer* writer = impl->createLSSerializer();
  if( writer->getDomConfig()->canSetParameter( XMLUni::fgDOMWRTFormatPrettyPrint, true ))
    writer->getDomConfig()->setParameter( XMLUni::fgDOMWRTFormatPrettyPrint, true );
  
  DOMDocumentType* doctype = impl->createDocumentType( cms::xerces::uStr("XML").ptr(), nullptr, nullptr );
  DOMDocument *    doc = 
    impl->createDocument( nullptr, cms::xerces::uStr(ADCToGeVConstant_tag.c_str()).ptr(), doctype );
    
  DOMElement* root = doc->getDocumentElement();
 
  xuti::writeHeader(root,header);
   
  xuti::WriteNodeWithValue(root,ESGain_tag,record.getESGain());

  std::string dump = cms::xerces::toString(writer->writeToString( root )); 
  doc->release();
  doctype->release();
  writer->release();

  return dump;
}

