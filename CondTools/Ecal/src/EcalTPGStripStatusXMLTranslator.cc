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

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "CondTools/Ecal/interface/EcalTPGStripStatusXMLTranslator.h"
#include "CondTools/Ecal/interface/DOMHelperFunctions.h"
#include "CondTools/Ecal/interface/XMLTags.h"

#include "CondFormats/DataRecord/interface/EcalTPGStripStatusRcd.h"

using namespace XERCES_CPP_NAMESPACE;
using namespace xuti;
using namespace std;

int  EcalTPGStripStatusXMLTranslator::readXML(const std::string& filename, 
					  EcalCondHeader& header,
					  EcalTPGStripStatus& record){

  std::cout << " TPGStripStatus should not be filled out from an xml file ..." << std::endl;
  cms::concurrency::xercesInitialize();

  XercesDOMParser* parser = new XercesDOMParser;
  parser->setValidationScheme( XercesDOMParser::Val_Never );
  parser->setDoNamespaces( false );
  parser->setDoSchema( false );
  
  parser->parse(filename.c_str());

  DOMDocument* xmlDoc = parser->getDocument();
  if (!xmlDoc) {
    std::cout << "EcalTPGStripStatusXMLTranslator::Error parsing document" << std::endl;
    return -1;
  }

  DOMElement* elementRoot = xmlDoc->getDocumentElement();

  xuti::readHeader(elementRoot,header);

  delete parser;
  cms::concurrency::xercesTerminate();
  return 0;
 }

int EcalTPGStripStatusXMLTranslator::writeXML(const std::string& filename, 
					  const EcalCondHeader& header,
					  const EcalTPGStripStatus& record){
  cms::concurrency::xercesInitialize();

  std::fstream fs(filename.c_str(),ios::out);
  fs<< dumpXML(header,record);

  cms::concurrency::xercesTerminate();

  return 0;  
}

std::string EcalTPGStripStatusXMLTranslator::dumpXML(const EcalCondHeader& header,const EcalTPGStripStatus& record){

  unique_ptr<DOMImplementation> impl( DOMImplementationRegistry::getDOMImplementation(cms::xerces::uStr("LS").ptr()));
  
  DOMLSSerializer* writer = impl->createLSSerializer();
  if( writer->getDomConfig()->canSetParameter( XMLUni::fgDOMWRTFormatPrettyPrint, true ))
    writer->getDomConfig()->setParameter( XMLUni::fgDOMWRTFormatPrettyPrint, true );
  
  DOMDocumentType* doctype = impl->createDocumentType( cms::xerces::uStr("XML").ptr(), nullptr, nullptr );
  DOMDocument* doc =
    impl->createDocument( nullptr, cms::xerces::uStr(TPGStripStatus_tag.c_str()).ptr(), doctype );
  DOMElement* root = doc->getDocumentElement();

  xuti::writeHeader(root,header);
  std::string TCC_tag("TCC");
  std::string TT_tag("TT");
  std::string ST_tag("ST");
  const EcalTPGStripStatusMap &stripMap = record.getMap();
  std::cout << "EcalTPGStripStatusXMLTranslator::dumpXML strip map size " << stripMap.size() << std::endl;
  EcalTPGStripStatusMapIterator itSt;
  for(itSt = stripMap.begin(); itSt != stripMap.end(); ++itSt) {
    if(itSt->second > 0) {
      int tccid = itSt->first/8192 & 0x7F;
      int tt  = itSt->first/64 & 0x7F;
      int pseudostrip  = itSt->first/8 & 0x7;
      //      std::cout << "Bad strip ID = " << itSt->first 
      //		<< " TCC " << tccid << " TT " << tt << " ST " << pseudostrip
      //		<< ", status = " << itSt->second << std::endl;
      DOMElement* cell_node = 
	root->getOwnerDocument()->createElement( cms::xerces::uStr(Cell_tag.c_str()).ptr());
      stringstream value_s;
      value_s << tccid ;
      cell_node->setAttribute(cms::xerces::uStr(TCC_tag.c_str()).ptr(),
			      cms::xerces::uStr(value_s.str().c_str()).ptr());
      value_s.str("");
      value_s << tt ;
      cell_node->setAttribute(cms::xerces::uStr(TT_tag.c_str()).ptr(),
			      cms::xerces::uStr(value_s.str().c_str()).ptr());
      value_s.str("");
      value_s << pseudostrip;
      cell_node->setAttribute(cms::xerces::uStr(ST_tag.c_str()).ptr(),
			      cms::xerces::uStr(value_s.str().c_str()).ptr());
      root->appendChild(cell_node);

      WriteNodeWithValue(cell_node, TPGStripStatus_tag, 1);
    }
  }

  std::string dump = cms::xerces::toString( writer->writeToString( root ));
  doc->release();
  doctype->release();
  writer->release();
  return dump;
}
