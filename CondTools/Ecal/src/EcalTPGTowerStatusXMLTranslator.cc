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
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

#include "CondTools/Ecal/interface/EcalTPGTowerStatusXMLTranslator.h"
#include "CondTools/Ecal/interface/DOMHelperFunctions.h"
#include "CondTools/Ecal/interface/XMLTags.h"

#include "CondFormats/DataRecord/interface/EcalTPGTowerStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTPGTowerStatus.h"

using namespace XERCES_CPP_NAMESPACE;
using namespace xuti;
using namespace std;

int  EcalTPGTowerStatusXMLTranslator::readXML(const std::string& filename, 
					  EcalCondHeader& header,
					  EcalTPGTowerStatus& record){

  std::cout << " TPGTowerStatus should not be filled out from an xml file ..." << std::endl;
  XMLPlatformUtils::Initialize();

  XercesDOMParser* parser = new XercesDOMParser;
  parser->setValidationScheme( XercesDOMParser::Val_Never );
  parser->setDoNamespaces( false );
  parser->setDoSchema( false );
  
  parser->parse(filename.c_str());

  DOMDocument* xmlDoc = parser->getDocument();
  if (!xmlDoc) {
    std::cout << "EcalTPGTowerStatusXMLTranslator::Error parsing document" << std::endl;
    return -1;
  }

  DOMElement* elementRoot = xmlDoc->getDocumentElement();

  xuti::readHeader(elementRoot,header);

  /*
  int chan = 0;
  while(cellnode) {
    int status = -1;
    DetId detid = readCellId(dynamic_cast<DOMElement*>(cellnode));

    DOMNode* my_node = getChildNode(cellnode,TPGTowerStatusCode_tag);
    GetNodeData(my_node, status);

    record[detid] = status;

    cellnode = cellnode->getNextSibling();

    while(cellnode && cellnode->getNodeType() != DOMNode::ELEMENT_NODE)
      cellnode = cellnode->getNextSibling();
    chan++;
  } 
  */
  delete parser;
  XMLPlatformUtils::Terminate();
  return 0;
 }

int EcalTPGTowerStatusXMLTranslator::writeXML(const std::string& filename, 
					  const EcalCondHeader& header,
					  const EcalTPGTowerStatus& record){
  std::fstream fs(filename.c_str(),ios::out);
  fs<< dumpXML(header,record);
  return 0;  
}


std::string EcalTPGTowerStatusXMLTranslator::dumpXML(const EcalCondHeader& header,const EcalTPGTowerStatus& record){

  XMLPlatformUtils::Initialize();
  DOMImplementation*  impl =
    DOMImplementationRegistry::getDOMImplementation(fromNative("LS").c_str());

  DOMWriter* writer =static_cast<DOMImplementationLS*>(impl)->createDOMWriter( );
  writer->setFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true);

  DOMDocumentType* doctype = impl->createDocumentType(fromNative("XML").c_str(), 0, 0 );
  DOMDocument *    doc = 
    impl->createDocument( 0, fromNative(TPGTowerStatus_tag).c_str(), doctype );

  doc->setEncoding(fromNative("UTF-8").c_str() );
  doc->setStandalone(true);
  doc->setVersion(fromNative("1.0").c_str() );

  DOMElement* root = doc->getDocumentElement();

  xuti::writeHeader(root,header);
  std::cout << "EcalTPGTowerStatusXMLTranslator::dumpXML" << std::endl;
  const EcalTPGTowerStatusMap &towerMap = record.getMap();
  std::cout << " tower map size " << towerMap.size() << std::endl;
  EcalTPGTowerStatusMapIterator it;
  for(it = towerMap.begin(); it != towerMap.end(); ++it) {
    if((*it).second > 0) {
      EcalTrigTowerDetId ttId((*it).first);
      std::cout << " TTDetId " << ttId << " eta " << ttId.ieta()  << " phi " << ttId.iphi() << std::endl;
      uint32_t rawid = ttId;
      DOMElement* cellnode=writeCell(root,rawid);
      WriteNodeWithValue(cellnode, TPGTowerStatus_tag, 1);
    }
  }

  std::string dump = toNative(writer->writeToString(*root)); 
  doc->release(); 
  return dump;
}

void EcalTPGTowerStatusXMLTranslator::plot(std::string fn, const EcalTPGTowerStatus& record){
  std::ofstream fout(fn.c_str());
  int valEB[34][72];
  for(int line = 0; line < 34; line++)
    for(int iphi = 0; iphi < 72; iphi++)
      valEB[line][iphi] = 0;

  const EcalTPGTowerStatusMap &towerMap = record.getMap();
  std::cout << " tower map size " << towerMap.size() << std::endl;
  EcalTPGTowerStatusMapIterator it;
  for(it = towerMap.begin(); it != towerMap.end(); ++it) {
    if((*it).second > 0) {
      EcalTrigTowerDetId ttId((*it).first);
      int ieta = ttId.ieta();
      int line = 17 - ieta;
      if(ieta < 0) line--;
      int iphi = ttId.iphi() - 1;  // 0 to 71
      valEB[line][iphi] = (*it).second;
    }
  }
  for(int line = 0; line < 34; line++) {
    for(int iphi = 0; iphi < 72; iphi++)
      fout << valEB[line][iphi] << " ";
    fout << std::endl;
    if(line == 16) fout << std::endl;
  }
  /*

  std::cout << " endcap size " << record.endcapItems().size() << std::endl;
  if (!record.endcapItems().size()) return;
  int valEE[2][20][20];
  for(int k = 0 ; k < 2; k++ ) 
    for(int ix = 0 ; ix < 20; ix++) 
      for(int iy = 0 ; iy < 20; iy++) 
	valEE[k][ix][iy] = -1;
  for(uint cellid = 0;
      cellid < EcalTrigTowerDetId::kEETotalTowers;
      ++cellid) {
    if(EcalScDetId::validHashIndex(cellid)) {
      EcalScDetId rawid = EcalScDetId::unhashIndex(cellid); 
      int ix = rawid.ix() - 1;  // 0 to 19
      int iy = 20 -  rawid.iy();  // 0 to 19
      int side = rawid.zside();
      int iz = side;
      if(side == -1) iz = 0;
      if(ix < 0 || ix > 19) std::cout << " Pb in ix " << ix << std::endl;
      if(iy < 0 || iy > 19) std::cout << " Pb in iy " << iy << std::endl;
      valEE[iz][ix][iy] = record[rawid].getStatusCode();
    }
  }
  for(int k = 0 ; k < 2; k++ ) {
    int iz = -1;
    if(k == 1) iz = 1;
    fout << " Side : " << iz << std::endl;
    for(int line = 0; line < 20; line++) {
      for(int ix = 0 ; ix < 20; ix++) {
	if(valEE[k][ix][line] < 0) fout << ". ";
	else fout << valEE[k][ix][line] << " ";
      }
      fout << std::endl;
    }
    fout << std::endl;
  }
  */
  return;
}
