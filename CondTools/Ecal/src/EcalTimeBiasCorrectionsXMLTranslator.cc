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

#include "CondFormats/EcalObjects/interface/EcalTimeBiasCorrections.h"
#include "CondTools/Ecal/interface/EcalTimeBiasCorrectionsXMLTranslator.h"
#include "CondTools/Ecal/interface/DOMHelperFunctions.h"

using namespace XERCES_CPP_NAMESPACE;
using namespace xuti;
using namespace std;

int  EcalTimeBiasCorrectionsXMLTranslator::readXML(const std::string& filename, 
					EcalCondHeader& header,
					EcalTimeBiasCorrections& record){

  cms::concurrency::xercesInitialize();

  XercesDOMParser* parser = new XercesDOMParser;
  parser->setValidationScheme( XercesDOMParser::Val_Never );
  parser->setDoNamespaces( false );
  parser->setDoSchema( false );
  
  parser->parse(filename.c_str());

  DOMDocument* xmlDoc = parser->getDocument();
  if (!xmlDoc) {
    std::cout << "EcalTimeBiasCorrectionsXMLTranslator::Error parsing document" << std::endl;
    return -1;
  }
  
  // Get the top-level element
  DOMElement* elementRoot = xmlDoc->getDocumentElement();

  xuti::readHeader(elementRoot, header);

  delete parser;
  cms::concurrency::xercesTerminate();
  return 0;
}

int EcalTimeBiasCorrectionsXMLTranslator::writeXML(const std::string& filename, 
						   const EcalCondHeader& header,
						   const EcalTimeBiasCorrections& record){
  cms::concurrency::xercesInitialize();

  std::fstream fs(filename.c_str(),ios::out);
  fs<< dumpXML(header,record);

  cms::concurrency::xercesTerminate();

  return 0;  
}

std::string EcalTimeBiasCorrectionsXMLTranslator::dumpXML(const EcalCondHeader& header,
							  const EcalTimeBiasCorrections& record){

  unique_ptr<DOMImplementation> impl( DOMImplementationRegistry::getDOMImplementation(cms::xerces::uStr("LS").ptr()));
  
  DOMLSSerializer* writer = impl->createLSSerializer();
  if( writer->getDomConfig()->canSetParameter( XMLUni::fgDOMWRTFormatPrettyPrint, true ))
    writer->getDomConfig()->setParameter( XMLUni::fgDOMWRTFormatPrettyPrint, true );
  
  DOMDocumentType* doctype = impl->createDocumentType( cms::xerces::uStr("XML").ptr(), nullptr, nullptr );
  DOMDocument* doc =
    impl->createDocument( nullptr, cms::xerces::uStr(IntercalibConstants_tag.c_str()).ptr(), doctype );
  DOMElement* root = doc->getDocumentElement();
 
  xuti::writeHeader(root,header);
   
  std::vector<float> vect = record.EBTimeCorrAmplitudeBins;
  std::vector<float>::iterator it;

  std::string ETCAB_tag = "EBTimeCorrAmplitudeBins";
  //  std::cout << ETCAB_tag << vect.size()<< "\n";
  DOMElement* ETCAB = 
    root->getOwnerDocument()->createElement( cms::xerces::uStr(ETCAB_tag.c_str()).ptr());
  root->appendChild(ETCAB);
  for (it = vect.begin(); it != vect.end(); it++ ) {
    //    std::cout << *it << " ";
    WriteNodeWithValue(ETCAB, Value_tag, *it);
  }
  //  std::cout << "\n";
  vect = record.EBTimeCorrShiftBins;
  ETCAB_tag = "EBTimeCorrShiftBins";
  //  std::cout << ETCAB_tag << vect.size()<< "\n";
  ETCAB = root->getOwnerDocument()->createElement( cms::xerces::uStr(ETCAB_tag.c_str()).ptr());
  root->appendChild(ETCAB);
  for (it = vect.begin(); it != vect.end(); it++ ) {
    //    std::cout << *it << " ";
    WriteNodeWithValue(ETCAB, Value_tag, *it);
  }
  //  std::cout << "\n";
  vect = record.EETimeCorrAmplitudeBins;
  ETCAB_tag = "EETimeCorrAmplitudeBins";
  //  std::cout << ETCAB_tag << vect.size()<< "\n";
  ETCAB = root->getOwnerDocument()->createElement( cms::xerces::uStr(ETCAB_tag.c_str()).ptr());
  root->appendChild(ETCAB);
  for (it = vect.begin(); it != vect.end(); it++ ) {
    //    std::cout << *it << " ";
    WriteNodeWithValue(ETCAB, Value_tag, *it);
  }
  //  std::cout << "\n";
  vect = record.EETimeCorrShiftBins;
  ETCAB_tag = "EETimeCorrShiftBins";
  //  std::cout << ETCAB_tag << vect.size()<< "\n";
  ETCAB = root->getOwnerDocument()->createElement( cms::xerces::uStr(ETCAB_tag.c_str()).ptr());
  root->appendChild(ETCAB);
  for (it = vect.begin(); it != vect.end(); it++ ) {
    //    std::cout << *it << " ";
    WriteNodeWithValue(ETCAB, Value_tag, *it);
  }
  //  std::cout << "\n";
 
  std::string dump = cms::xerces::toString( writer->writeToString( root ));
  doc->release();
  doctype->release();
  writer->release();

  return dump;
}

