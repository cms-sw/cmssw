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

#include "CondTools/Ecal/interface/EcalClusterEnergyCorrectionObjectSpecificXMLTranslator.h"
#include "CondTools/Ecal/interface/DOMHelperFunctions.h"

using namespace XERCES_CPP_NAMESPACE;
using namespace xuti;
using namespace std;


int  
EcalClusterEnergyCorrectionObjectSpecificXMLTranslator::readXML(
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
    std::cout << "EcalClusterEnergyCorrectionObjectSpecificXMLTranslator::Error parsing document" 
	      << std::endl;
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
EcalClusterEnergyCorrectionObjectSpecificXMLTranslator::dumpXML(       
		       const EcalCondHeader& header,
		       const EcalFunParams& record) {
    
  unique_ptr<DOMImplementation> impl( DOMImplementationRegistry::getDOMImplementation( cms::xerces::uStr("LS").ptr()));
  
  DOMLSSerializer* writer = impl->createLSSerializer();
  if( writer->getDomConfig()->canSetParameter( XMLUni::fgDOMWRTFormatPrettyPrint, true ))
    writer->getDomConfig()->setParameter( XMLUni::fgDOMWRTFormatPrettyPrint, true );
  
  DOMDocumentType* doctype = 
    impl->createDocumentType( cms::xerces::uStr("XML").ptr(), nullptr, nullptr );
  const  std::string ECECOS_tag("EcalClusterEnergyCorrectionObjectSpecific");
  DOMDocument *    doc = 
    impl->createDocument( nullptr, cms::xerces::uStr(ECECOS_tag.c_str()).ptr(), doctype );
  
  DOMElement* root = doc->getDocumentElement();
  xuti::writeHeader(root, header);

  const std::string ECEC_tag[9] = {"fEta","fBremEtaElectrons","fBremEtaPhotons",
				   "fEtElectronsEB","fEtElectronsEE","fEtPhotonsEB","fEtPhotonsEE",
				   "fEnergyElectronsEE","fEnergyPhotonsEE"};
  int tit = 0;
  int par = 0;
  for ( EcalFunctionParameters::const_iterator it = record.params().begin(); it != record.params().end(); ++it ) {
    if(par < 2) tit = 0;
    else if(par < 86) tit = 1;
    else if(par < 170) tit = 2;
    else if(par < 177) tit = 3;
    else if(par < 184) tit = 4;
    else if(par < 191) tit = 5;
    else if(par < 198) tit = 6;
    else if(par < 203) tit = 7;
    else tit = 8;
    DOMElement* ECEC = 
      root->getOwnerDocument()->createElement( cms::xerces::uStr(ECEC_tag[tit].c_str()).ptr());
    root->appendChild(ECEC);

    WriteNodeWithValue(ECEC,Value_tag,*it);
    par++;
  }
  std::cout << "\n";
 
  std::string dump = cms::xerces::toString(writer->writeToString( root )); 
  doc->release();
  doctype->release();
  writer->release();

  return dump;
}

int 
EcalClusterEnergyCorrectionObjectSpecificXMLTranslator::writeXML(
               const std::string& filename,         
	       const EcalCondHeader&   header,
	       const EcalFunParams& record) {

  cms::concurrency::xercesInitialize();

  std::fstream fs(filename.c_str(),ios::out);
  fs<< dumpXML(header,record);

  cms::concurrency::xercesTerminate();

  return 0;  
}
