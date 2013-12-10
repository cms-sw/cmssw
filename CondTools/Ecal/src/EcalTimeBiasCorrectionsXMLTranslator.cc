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
  std::fstream fs(filename.c_str(),ios::out);
  fs<< dumpXML(header,record);
  return 0;  
}

std::string EcalTimeBiasCorrectionsXMLTranslator::dumpXML(const EcalCondHeader& header,
							  const EcalTimeBiasCorrections& record){

  cms::concurrency::xercesInitialize();
  
  DOMImplementation*  impl =
    DOMImplementationRegistry::getDOMImplementation(fromNative("LS").c_str());
  
  DOMWriter* writer =static_cast<DOMImplementationLS*>(impl)->createDOMWriter( );
  writer->setFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true);
  
  DOMDocumentType* doctype = impl->createDocumentType(fromNative("XML").c_str(), 0, 0 );
  DOMDocument *    doc = 
    impl->createDocument( 0, fromNative(IntercalibConstants_tag).c_str(), doctype );


  doc->setEncoding(fromNative("UTF-8").c_str() );
  doc->setStandalone(true);
  doc->setVersion(fromNative("1.0").c_str() );
    
  DOMElement* root = doc->getDocumentElement();
 
  xuti::writeHeader(root,header);
   
  std::vector<float> vect = record.EBTimeCorrAmplitudeBins;
  std::vector<float>::iterator it;

  std::string ETCAB_tag = "EBTimeCorrAmplitudeBins";
  //  std::cout << ETCAB_tag << vect.size()<< "\n";
  DOMElement* ETCAB = 
    root->getOwnerDocument()->createElement( fromNative(ETCAB_tag).c_str());
  root->appendChild(ETCAB);
  for (it = vect.begin(); it != vect.end(); it++ ) {
    //    std::cout << *it << " ";
    WriteNodeWithValue(ETCAB, Value_tag, *it);
  }
  //  std::cout << "\n";
  vect = record.EBTimeCorrShiftBins;
  ETCAB_tag = "EBTimeCorrShiftBins";
  //  std::cout << ETCAB_tag << vect.size()<< "\n";
  ETCAB = root->getOwnerDocument()->createElement( fromNative(ETCAB_tag).c_str());
  root->appendChild(ETCAB);
  for (it = vect.begin(); it != vect.end(); it++ ) {
    //    std::cout << *it << " ";
    WriteNodeWithValue(ETCAB, Value_tag, *it);
  }
  //  std::cout << "\n";
  vect = record.EETimeCorrAmplitudeBins;
  ETCAB_tag = "EETimeCorrAmplitudeBins";
  //  std::cout << ETCAB_tag << vect.size()<< "\n";
  ETCAB = root->getOwnerDocument()->createElement( fromNative(ETCAB_tag).c_str());
  root->appendChild(ETCAB);
  for (it = vect.begin(); it != vect.end(); it++ ) {
    //    std::cout << *it << " ";
    WriteNodeWithValue(ETCAB, Value_tag, *it);
  }
  //  std::cout << "\n";
  vect = record.EETimeCorrShiftBins;
  ETCAB_tag = "EETimeCorrShiftBins";
  //  std::cout << ETCAB_tag << vect.size()<< "\n";
  ETCAB = root->getOwnerDocument()->createElement( fromNative(ETCAB_tag).c_str());
  root->appendChild(ETCAB);
  for (it = vect.begin(); it != vect.end(); it++ ) {
    //    std::cout << *it << " ";
    WriteNodeWithValue(ETCAB, Value_tag, *it);
  }
  //  std::cout << "\n";
 
  std::string dump= toNative(writer->writeToString(*root));
  doc->release();
  return dump;
}

