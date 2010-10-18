#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <xercesc/dom/DOMNode.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/sax/SAXException.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>

#include "CondTools/Ecal/interface/EcalDCSTowerStatusXMLTranslator.h"
#include "CondTools/Ecal/interface/DOMHelperFunctions.h"
#include "CondTools/Ecal/interface/XMLTags.h"

using namespace XERCES_CPP_NAMESPACE;
using namespace xuti;
using namespace std;

int  EcalDCSTowerStatusXMLTranslator::readXML(const std::string& filename, 
					  EcalCondHeader& header,
					  EcalDCSTowerStatus& record){

  std::cout << " DCSTowerStatus should not be filled out from an xml file ..." << std::endl;
  XMLPlatformUtils::Initialize();

  XercesDOMParser* parser = new XercesDOMParser;
  parser->setValidationScheme( XercesDOMParser::Val_Never );
  parser->setDoNamespaces( false );
  parser->setDoSchema( false );
  
  parser->parse(filename.c_str());

  DOMDocument* xmlDoc = parser->getDocument();
  if (!xmlDoc) {
    std::cout << "EcalDCSTowerStatusXMLTranslator::Error parsing document" << std::endl;
    return -1;
  }

  DOMElement* elementRoot = xmlDoc->getDocumentElement();

  xuti::readHeader(elementRoot,header);

  DOMNode * cellnode = getChildNode(elementRoot,Cell_tag);

  int chan = 0;
  while(cellnode) {
    int status = -1;
    DetId detid = readCellId(dynamic_cast<DOMElement*>(cellnode));

    DOMNode* my_node = getChildNode(cellnode,DCSStatusCode_tag);
    GetNodeData(my_node, status);

    record[detid] = status;

    cellnode = cellnode->getNextSibling();

    while(cellnode && cellnode->getNodeType() != DOMNode::ELEMENT_NODE)
      cellnode = cellnode->getNextSibling();
    chan++;
  } 

  delete parser;
  XMLPlatformUtils::Terminate();
  return 0;
 }

int EcalDCSTowerStatusXMLTranslator::writeXML(const std::string& filename, 
					  const EcalCondHeader& header,
					  const EcalDCSTowerStatus& record){
  std::fstream fs(filename.c_str(),ios::out);
  fs<< dumpXML(header,record);
  return 0;  
}


std::string EcalDCSTowerStatusXMLTranslator::dumpXML(const EcalCondHeader& header,const EcalDCSTowerStatus& record){

  XMLPlatformUtils::Initialize();
  DOMImplementation*  impl =
    DOMImplementationRegistry::getDOMImplementation(fromNative("LS").c_str());

  DOMWriter* writer =static_cast<DOMImplementationLS*>(impl)->createDOMWriter( );
  writer->setFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true);

  DOMDocumentType* doctype = impl->createDocumentType(fromNative("XML").c_str(), 0, 0 );
  DOMDocument *    doc = 
    impl->createDocument( 0, fromNative(DCSTowerStatus_tag).c_str(), doctype );

  doc->setEncoding(fromNative("UTF-8").c_str() );
  doc->setStandalone(true);
  doc->setVersion(fromNative("1.0").c_str() );

  DOMElement* root = doc->getDocumentElement();

  xuti::writeHeader(root,header);
  std::cout << " barrel size " << record.barrelItems().size() << std::endl;
  if (!record.barrelItems().size()) return std::string();
  for(uint cellid = 0;
      cellid < EcalTrigTowerDetId::kEBTotalTowers;
      ++cellid) {
    uint32_t rawid = EcalTrigTowerDetId::detIdFromDenseIndex(cellid);
    if (record.find(rawid) == record.end()) continue;
    DOMElement* cellnode=writeCell(root,rawid);

    WriteNodeWithValue(cellnode, DCSStatusCode_tag, record[rawid].getStatusCode());
  } 

  std::cout << " endcap size " << record.endcapItems().size() << std::endl;
  if (!record.endcapItems().size()) return std::string();
  for(uint cellid = 0;
      cellid < EcalTrigTowerDetId::kEETotalTowers;
      ++cellid) {
    if(!EcalScDetId::validHashIndex(cellid)) continue;
    uint32_t rawid = EcalScDetId::unhashIndex(cellid); 

    if (record.find(rawid) == record.end()) continue;
    DOMElement* cellnode=writeCell(root,rawid);

    WriteNodeWithValue(cellnode, DCSStatusCode_tag, record[rawid].getStatusCode());
  }

  std::string dump = toNative(writer->writeToString(*root)); 
  doc->release(); 
  return dump;
}

void EcalDCSTowerStatusXMLTranslator::plot(std::string fn, const EcalDCSTowerStatus& record){
  std::ofstream fout(fn.c_str());
  int valEB[34][72];
  std::cout << " barrel size " << record.barrelItems().size() << std::endl;
  if (!record.barrelItems().size()) return;
  for(uint cellid = 0;
      cellid < EcalTrigTowerDetId::kEBTotalTowers;
      ++cellid) {
    EcalTrigTowerDetId rawid = EcalTrigTowerDetId::detIdFromDenseIndex(cellid);
    if (record.find(rawid) == record.end()) continue;
    int ieta = rawid.ieta();
    int line = 17 - ieta;
    if(ieta < 0) line--;
    int iphi = rawid.iphi() - 1;  // 0 to 71
    valEB[line][iphi] = record[rawid].getStatusCode();
  }
  for(int line = 0; line < 34; line++) {
    for(int iphi = 0; iphi < 72; iphi++)
      fout << valEB[line][iphi] << " ";
    fout << std::endl;
    if(line == 16) fout << std::endl;
  }

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
	if(valEE[k][ix][line] < 0) fout << " . ";
	else fout << setw(2) << valEE[k][ix][line] << " ";
      }
      fout << std::endl;
    }
    fout << std::endl;
  }

  return;
}
