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

#include "CondTools/Ecal/interface/EcalClusterCrackCorrXMLTranslator.h"
#include "CondTools/Ecal/interface/DOMHelperFunctions.h"

using namespace XERCES_CPP_NAMESPACE;
using namespace xuti;
using namespace std;


int  
EcalClusterCrackCorrXMLTranslator::readXML(
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
    std::cout << "EcalClusterCrackCorrXMLTranslator::Error parsing document" << std::endl;
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
EcalClusterCrackCorrXMLTranslator::dumpXML(       
		       const EcalCondHeader&   header,
		       const EcalFunParams& record) {
  
  unique_ptr<DOMImplementation> impl( DOMImplementationRegistry::getDOMImplementation( cms::xerces::uStr("LS").ptr()));
  
  DOMLSSerializer* writer = impl->createLSSerializer();
  if( writer->getDomConfig()->canSetParameter( XMLUni::fgDOMWRTFormatPrettyPrint, true ))
    writer->getDomConfig()->setParameter( XMLUni::fgDOMWRTFormatPrettyPrint, true );
  
  DOMDocumentType* doctype = 
    impl->createDocumentType( cms::xerces::uStr("XML").ptr(), nullptr, nullptr );
  const  std::string EcalClusterCrackCorr_tag("EcalClusterCrackCorr");
  DOMDocument *    doc = 
    impl->createDocument( nullptr, cms::xerces::uStr(EcalClusterCrackCorr_tag.c_str()).ptr(), doctype );
    
  DOMElement* root = doc->getDocumentElement();
  xuti::writeHeader(root, header);

  const std::string ECCC_tag[4] = {"IPCloseEtaSide", "IPFarEtaSide",
				   "IPClosePhiSide", "IPFarPhiSide"};;
  int num = 0;
  for ( EcalFunctionParameters::const_iterator it = record.params().begin(); it != record.params().end(); ++it ) {
    int side = num /5;
    int par = num%5;
    std::string s;
    std::stringstream out;
    out << par;
    s = out.str();
    std::string sw = ECCC_tag[side] + "_" + s;
    DOMElement* ECCC = 
      root->getOwnerDocument()->createElement( cms::xerces::uStr(sw.c_str()).ptr());
    root->appendChild(ECCC);

    WriteNodeWithValue(ECCC,Value_tag,*it);
    num++;
  } 
  
  std::string dump = cms::xerces::toString(writer->writeToString( root )); 
  doc->release();
  doctype->release();
  writer->release();

  return dump;
}

int 
EcalClusterCrackCorrXMLTranslator::writeXML(
               const std::string& filename,         
	       const EcalCondHeader&   header,
	       const EcalFunParams& record) {

  cms::concurrency::xercesInitialize();

  std::fstream fs(filename.c_str(),ios::out);
  fs<< dumpXML(header,record);

  cms::concurrency::xercesTerminate();

  return 0;  
}
