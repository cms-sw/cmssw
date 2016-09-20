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

#include "CondTools/Ecal/interface/EcalClusterEnergyCorrectionXMLTranslator.h"
#include "CondTools/Ecal/interface/DOMHelperFunctions.h"

using namespace XERCES_CPP_NAMESPACE;
using namespace xuti;
using namespace std;


int  
EcalClusterEnergyCorrectionXMLTranslator::readXML(
           const string& filename,
	   EcalCondHeader& header,
	   EcalFunParams& record){

  cms::concurrency::xercesInitialize();

  XercesDOMParser* parser = new XercesDOMParser;
  parser->setValidationScheme( XercesDOMParser::Val_Never );
  parser->setDoNamespaces( false );
  parser->setDoSchema( false );
  
  parser->parse(filename.c_str());

  DOMDocument* xmlDoc = parser->getDocument();
  

  if (!xmlDoc) {
    std::cout << "EcalClusterEnergyCorrectionXMLTranslator::Error parsing document" << std::endl;
    return -1;
  }

  DOMElement* elementRoot = xmlDoc->getDocumentElement();

  xuti::readHeader(elementRoot, header);

  // need some extra code here

  delete parser;
  cms::concurrency::xercesTerminate();
  return 0;
}

std::string 
EcalClusterEnergyCorrectionXMLTranslator::dumpXML(       
		       const EcalCondHeader&   header,
		       const EcalFunParams& record){
  
  cms::concurrency::xercesInitialize();
  
  unique_ptr<DOMImplementation> impl( DOMImplementationRegistry::getDOMImplementation(fromNative("LS").c_str()));
  
  DOMLSSerializer* writer = impl->createLSSerializer();
  if( writer->getDomConfig()->canSetParameter( XMLUni::fgDOMWRTFormatPrettyPrint, true ))
    writer->getDomConfig()->setParameter( XMLUni::fgDOMWRTFormatPrettyPrint, true );
  
  DOMDocumentType* doctype = 
    impl->createDocumentType( fromNative("XML").c_str(), 0, 0 );
  const  std::string EcalClusterEnergyCorrection_tag("EcalClusterEnergyCorrection");
  DOMDocument *    doc = 
    impl->createDocument( 0, fromNative(EcalClusterEnergyCorrection_tag).c_str(), doctype );
  
  DOMElement* root = doc->getDocumentElement();
  xuti::writeHeader(root, header);

  const std::string ECEC_tag("ClusterEnergy");
  for ( EcalFunctionParameters::const_iterator it = record.params().begin(); it != record.params().end(); ++it ) {
    DOMElement* ECEC = 
      root->getOwnerDocument()->createElement( fromNative(ECEC_tag).c_str());
    root->appendChild(ECEC);

    WriteNodeWithValue(ECEC,Value_tag,*it);
  }
  
  std::string dump = toNative(writer->writeToString( root )); 
  doc->release();
  doctype->release();
  writer->release();

  return dump;
}

int 
EcalClusterEnergyCorrectionXMLTranslator::writeXML(
               const std::string& filename,         
	       const EcalCondHeader&   header,
	       const EcalFunParams& record){

  std::fstream fs(filename.c_str(),ios::out);
  fs<< dumpXML(header,record);
  return 0;  
}
