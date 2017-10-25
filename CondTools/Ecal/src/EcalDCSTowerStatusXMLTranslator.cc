#include <iostream>
#include <iomanip>
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
  cms::concurrency::xercesInitialize();

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
  cms::concurrency::xercesTerminate();
  return 0;
 }

int EcalDCSTowerStatusXMLTranslator::writeXML(const std::string& filename, 
					  const EcalCondHeader& header,
					  const EcalDCSTowerStatus& record){
  cms::concurrency::xercesInitialize();

  std::fstream fs(filename.c_str(),ios::out);
  fs<< dumpXML(header,record);

  cms::concurrency::xercesTerminate();

  return 0;  
}

std::string EcalDCSTowerStatusXMLTranslator::dumpXML(const EcalCondHeader& header,const EcalDCSTowerStatus& record){

  unique_ptr<DOMImplementation> impl( DOMImplementationRegistry::getDOMImplementation(cms::xerces::uStr("LS").ptr()));
  
  DOMLSSerializer* writer = impl->createLSSerializer();
  if( writer->getDomConfig()->canSetParameter( XMLUni::fgDOMWRTFormatPrettyPrint, true ))
    writer->getDomConfig()->setParameter( XMLUni::fgDOMWRTFormatPrettyPrint, true );
  
  DOMDocumentType* doctype = impl->createDocumentType( cms::xerces::uStr("XML").ptr(), nullptr, nullptr );
  DOMDocument* doc = impl->createDocument( nullptr, cms::xerces::uStr(DCSTowerStatus_tag.c_str()).ptr(), doctype );
  DOMElement* root = doc->getDocumentElement();

  xuti::writeHeader(root,header);
  std::cout << " barrel size " << record.barrelItems().size() << std::endl;
  if (record.barrelItems().empty()) return std::string();
  for(uint cellid = 0;
      cellid < EcalTrigTowerDetId::kEBTotalTowers;
      ++cellid) {
    uint32_t rawid = EcalTrigTowerDetId::detIdFromDenseIndex(cellid);
    if (record.find(rawid) == record.end()) continue;
    DOMElement* cellnode=writeCell(root,rawid);

    WriteNodeWithValue(cellnode, DCSStatusCode_tag, record[rawid].getStatusCode());
  } 

  std::cout << " endcap size " << record.endcapItems().size() << std::endl;
  if (record.endcapItems().empty()) return std::string();
  for(uint cellid = 0;
      cellid < EcalTrigTowerDetId::kEETotalTowers;
      ++cellid) {
    if(!EcalScDetId::validHashIndex(cellid)) continue;
    uint32_t rawid = EcalScDetId::unhashIndex(cellid); 

    if (record.find(rawid) == record.end()) continue;
    DOMElement* cellnode=writeCell(root,rawid);

    WriteNodeWithValue(cellnode, DCSStatusCode_tag, record[rawid].getStatusCode());
  }

  std::string dump = cms::xerces::toString( writer->writeToString( root ));
  doc->release();
  doctype->release();
  writer->release();

  return dump;
}

void EcalDCSTowerStatusXMLTranslator::plot(std::string fn, const EcalDCSTowerStatus& record){
  std::ofstream fout(fn.c_str());
  int valEB[34][72];
  std::cout << " barrel size " << record.barrelItems().size() << std::endl;
  if (record.barrelItems().empty()) return;
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
  if (record.endcapItems().empty()) return;
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
