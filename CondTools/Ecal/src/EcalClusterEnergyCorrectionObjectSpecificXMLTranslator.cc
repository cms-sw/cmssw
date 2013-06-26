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

  XMLPlatformUtils::Initialize();

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
  XMLPlatformUtils::Terminate();
  return 0;
}

std::string 
EcalClusterEnergyCorrectionObjectSpecificXMLTranslator::dumpXML(       
		       const EcalCondHeader&   header,
		       const EcalFunParams& record){
  
  XMLPlatformUtils::Initialize();
  
  DOMImplementation*  impl =
    DOMImplementationRegistry::getDOMImplementation(fromNative("LS").c_str());
  
  DOMWriter* writer =
    static_cast<DOMImplementationLS*>(impl)->createDOMWriter( );
  writer->setFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true);
  
  DOMDocumentType* doctype = 
    impl->createDocumentType( fromNative("XML").c_str(), 0, 0 );
  const  std::string ECECOS_tag("EcalClusterEnergyCorrectionObjectSpecific");
  DOMDocument *    doc = 
    impl->createDocument( 0, fromNative(ECECOS_tag).c_str(), doctype );
  
  
  doc->setEncoding(fromNative("UTF-8").c_str() );
  doc->setStandalone(true);
  doc->setVersion(fromNative("1.0").c_str() );
  
  
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
      root->getOwnerDocument()->createElement( fromNative(ECEC_tag[tit]).c_str());
    root->appendChild(ECEC);

    WriteNodeWithValue(ECEC,Value_tag,*it);
    par++;
  }
  std::cout << "\n";
 
  std::string dump= toNative(writer->writeToString(*root));
  doc->release();
  return dump;
}

int 
EcalClusterEnergyCorrectionObjectSpecificXMLTranslator::writeXML(
               const std::string& filename,         
	       const EcalCondHeader&   header,
	       const EcalFunParams& record){

  std::fstream fs(filename.c_str(),ios::out);
  fs<< dumpXML(header,record);
  return 0;  
}
