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

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "CondTools/Ecal/interface/EcalTPGCrystalStatusXMLTranslator.h"
#include "CondTools/Ecal/interface/DOMHelperFunctions.h"
#include "CondTools/Ecal/interface/XMLTags.h"

#include "CondFormats/DataRecord/interface/EcalTPGCrystalStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h"

using namespace XERCES_CPP_NAMESPACE;
using namespace xuti;
using namespace std;

int EcalTPGCrystalStatusXMLTranslator::writeXML(const std::string& filename, 
						const EcalCondHeader& header,
						const EcalTPGCrystalStatus& record){
  std::fstream fs(filename.c_str(),ios::out);
  fs<< dumpXML(header,record);
  return 0;  
}

std::string EcalTPGCrystalStatusXMLTranslator::dumpXML(const EcalCondHeader& header,const EcalTPGCrystalStatus& record){

  XMLPlatformUtils::Initialize();
  DOMImplementation*  impl =
    DOMImplementationRegistry::getDOMImplementation(fromNative("LS").c_str());

  DOMWriter* writer =static_cast<DOMImplementationLS*>(impl)->createDOMWriter( );
  writer->setFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true);

  DOMDocumentType* doctype = impl->createDocumentType(fromNative("XML").c_str(), 0, 0 );
  DOMDocument *    doc = 
    impl->createDocument( 0, fromNative(TPGCrystalStatus_tag).c_str(), doctype );

  doc->setEncoding(fromNative("UTF-8").c_str() );
  doc->setStandalone(true);
  doc->setVersion(fromNative("1.0").c_str() );

  DOMElement* root = doc->getDocumentElement();

  xuti::writeHeader(root,header);

  const int kSides       = 2;
  const int kBarlRings   = EBDetId::MAX_IETA;
  const int kBarlWedges  = EBDetId::MAX_IPHI;
  const int kEndcWedgesX = EEDetId::IX_MAX;
  const int kEndcWedgesY = EEDetId::IY_MAX;

  std::cout << "EcalTPGCrystalStatusXMLTranslator::dumpXML" << std::endl;
  for (int sign=0; sign < kSides; sign++) {
    int thesign = sign==1 ? 1:-1;

    for (int ieta=0; ieta<kBarlRings; ieta++) {
      for (int iphi=0; iphi<kBarlWedges; iphi++) {
	EBDetId id((ieta+1)*thesign, iphi+1);
	if(record[id.rawId()].getStatusCode() > 0) {
	  DOMElement* cellnode=writeCell(root, id);
	  WriteNodeWithValue(cellnode, TPGCrystalStatus_tag, record[id.rawId()].getStatusCode());
	}
      }  // iphi
    }   // ieta

    for (int ix=0; ix<kEndcWedgesX; ix++) {
      for (int iy=0; iy<kEndcWedgesY; iy++) {
	if (! EEDetId::validDetId(ix+1,iy+1,thesign)) continue;
	EEDetId id(ix+1,iy+1,thesign);
	if(record[id.rawId()].getStatusCode() > 0) {
	  DOMElement* cellnode=writeCell(root, id);
	  WriteNodeWithValue(cellnode, TPGCrystalStatus_tag, record[id.rawId()].getStatusCode());
	}
      }  // iy
    }   // ix
  }    // side

  std::string dump = toNative(writer->writeToString(*root)); 
  doc->release(); 
  return dump;
}
