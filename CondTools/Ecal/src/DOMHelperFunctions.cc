/**
 *  \file Implementation of helper functions
 *
 *  $Id: DOMHelperFunctions.cc,v 1.5 2010/12/16 08:54:06 fay Exp $
 */


#include "CondTools/Ecal/interface/DOMHelperFunctions.h"
#include "CondTools/Ecal/interface/XMLTags.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include <xercesc/dom/DOM.hpp>
#include <xercesc/dom/DOMNode.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include "FWCore/Concurrency/interface/Xerces.h"
#include "Utilities/Xerces/interface/XercesStrUtils.h"
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <sstream>

using namespace std;
using namespace xuti;
using namespace xercesc;

const DetId xuti::readCellId(xercesc::DOMElement* node){
  

  int ieta =0;
  int iphi =0;
  int ix   =0;
  int iy   =0;
  int ixSC =0;
  int iySC =0;
  int zside=0;
  
  stringstream ieta_str ;
  stringstream iphi_str;
  stringstream ix_str;
  stringstream iy_str ;
  stringstream ixSC_str;
  stringstream iySC_str ;
  stringstream zside_str ;

 
  ieta_str << cms::xerces::toString(node->getAttribute(cms::xerces::uStr(iEta_tag.c_str()).ptr()));
  iphi_str << cms::xerces::toString(node->getAttribute(cms::xerces::uStr(iPhi_tag.c_str()).ptr()));
  ix_str   << cms::xerces::toString(node->getAttribute(cms::xerces::uStr(ix_tag.c_str()).ptr()));
  iy_str   << cms::xerces::toString(node->getAttribute(cms::xerces::uStr(iy_tag.c_str()).ptr()));
  ixSC_str << cms::xerces::toString(node->getAttribute(cms::xerces::uStr(ixSC_tag.c_str()).ptr()));
  iySC_str << cms::xerces::toString(node->getAttribute(cms::xerces::uStr(iySC_tag.c_str()).ptr()));
  zside_str<< cms::xerces::toString(node->getAttribute(cms::xerces::uStr(zside_tag.c_str()).ptr()));
  
  ieta_str>> ieta;
  iphi_str>> iphi;
  ix_str  >> ix;
  iy_str  >> iy;
  ixSC_str >> ixSC;
  iySC_str >> iySC;
  zside_str >> zside;

  if (ieta && iphi)        {return EBDetId(ieta,iphi);}
  if (ix   && iy  && zside){return EEDetId(ix,iy,zside);}
  if (ixSC && iySC && zside){return EcalScDetId(ixSC, iySC, zside);}
  
  cerr<<"XMLCell: error reading cell, missing field ?"<<std::endl;
  return 0;
 
}



DOMElement*  xuti::writeCell(xercesc::DOMNode* node, 
		  const DetId& detid){
  
  DOMElement* cell_node = 
      node->getOwnerDocument()->createElement( cms::xerces::uStr(Cell_tag.c_str()).ptr());

  node->appendChild(cell_node);
  
  if (detid.subdetId() == EcalBarrel ){ 
   
    stringstream value_s;
    value_s <<EBDetId(detid).ieta() ;

    cell_node->setAttribute(cms::xerces::uStr(iEta_tag.c_str()).ptr(),
			    cms::xerces::uStr(value_s.str().c_str()).ptr());
    value_s.str("");
    value_s <<EBDetId(detid).iphi() ;

    cell_node->setAttribute(cms::xerces::uStr(iPhi_tag.c_str()).ptr(),
			    cms::xerces::uStr(value_s.str().c_str()).ptr());

  } else if (detid.subdetId() == EcalEndcap){
    
    // is it a EcalScDetId ?
    unsigned int ScIdCheck = detid.rawId() & 0x00008000;
    if (ScIdCheck == 0) {
      stringstream value_s;
      value_s <<EEDetId(detid).ix() ;
  
      cell_node->setAttribute(cms::xerces::uStr(ix_tag.c_str()).ptr(),
			      cms::xerces::uStr(value_s.str().c_str()).ptr());
      value_s.str("");
      value_s <<EEDetId(detid).iy() ;

      cell_node->setAttribute(cms::xerces::uStr(iy_tag.c_str()).ptr(),
			      cms::xerces::uStr(value_s.str().c_str()).ptr());
      value_s.str("");
      value_s <<EEDetId(detid).zside() ;

      cell_node->setAttribute(cms::xerces::uStr(zside_tag.c_str()).ptr(),
			      cms::xerces::uStr(value_s.str().c_str()).ptr());
    }
    else {
      stringstream value_s;
      value_s << EcalScDetId(detid).ix() ;
  
      cell_node->setAttribute(cms::xerces::uStr(ixSC_tag.c_str()).ptr(),
			      cms::xerces::uStr(value_s.str().c_str()).ptr());
      value_s.str("");
      value_s << EcalScDetId(detid).iy() ;

      cell_node->setAttribute(cms::xerces::uStr(iySC_tag.c_str()).ptr(),
			      cms::xerces::uStr(value_s.str().c_str()).ptr());
      value_s.str("");
      value_s << EcalScDetId(detid).zside() ;

      cell_node->setAttribute(cms::xerces::uStr(zside_tag.c_str()).ptr(),
			      cms::xerces::uStr(value_s.str().c_str()).ptr());
    }

  } else if (detid.subdetId() == EcalTriggerTower ){ 
    stringstream value_s;
    value_s <<EcalTrigTowerDetId(detid).ieta() ;
  
    cell_node->setAttribute(cms::xerces::uStr(iEta_tag.c_str()).ptr(),
			    cms::xerces::uStr(value_s.str().c_str()).ptr());
    value_s.str("");
    value_s <<EcalTrigTowerDetId(detid).iphi() ;

    cell_node->setAttribute(cms::xerces::uStr(iPhi_tag.c_str()).ptr(),
			    cms::xerces::uStr(value_s.str().c_str()).ptr());
  }
  return cell_node;
}
  
// return 0 if not found
DOMNode * xuti::getChildNode(DOMNode * node,  const std::string& nodename ){

   if (!node) return nullptr;

  for (DOMNode* childNode = node->getFirstChild(); 
       childNode; childNode = childNode->getNextSibling()) {
    
    if (childNode->getNodeType() == DOMNode::ELEMENT_NODE) {
      
      const string foundName = cms::xerces::toString(childNode->getNodeName());
      
      if (foundName == nodename) return childNode;
    }// if element 
  }// for child 
  
  return nullptr;

}



void xuti::writeHeader (xercesc::DOMNode* parentNode, 
			const EcalCondHeader& header ){


  
  DOMElement* headernode =
    parentNode->getOwnerDocument()->createElement( cms::xerces::uStr(Header_tag.c_str()).ptr());
  parentNode->appendChild(headernode);

  // write the actual header
  WriteNodeWithValue(headernode, Header_methodtag, header.method_);
  WriteNodeWithValue(headernode, Header_versiontag, header.version_);
  WriteNodeWithValue(headernode, Header_datasourcetag, header.datasource_);
  WriteNodeWithValue(headernode, Header_sincetag, header.since_);
  WriteNodeWithValue(headernode, Header_tagtag, header.tag_);
  WriteNodeWithValue(headernode, Header_datetag, header.date_);

}


void xuti::readHeader(xercesc::DOMNode* parentNode, 
		      EcalCondHeader& header){

  DOMNode * hnode = getChildNode(parentNode,Header_tag);

  DOMNode * node  = getChildNode(hnode,Header_methodtag);
  GetNodeStringData(node,header.method_);

  node  = getChildNode(hnode,Header_versiontag);
  GetNodeStringData(node,header.version_);

  node  = getChildNode(hnode,Header_datasourcetag);
  GetNodeStringData(node,header.datasource_);

  node  = getChildNode(hnode,Header_sincetag);
  GetNodeData(node,header.since_);

  node  = getChildNode(hnode,Header_tagtag);
  GetNodeStringData(node,header.tag_);

  
  node  = getChildNode(hnode,Header_datetag);
  GetNodeStringData(node,header.date_);

}

int xuti::readHeader(const std::string& filename,EcalCondHeader& header ){
  
  cms::concurrency::xercesInitialize();

  XercesDOMParser* parser = new XercesDOMParser;
  parser->setValidationScheme( XercesDOMParser::Val_Never );
  parser->setDoNamespaces( false );
  parser->setDoSchema( false );
  
  parser->parse(filename.c_str());

  DOMDocument* xmlDoc = parser->getDocument();
  

  if (!xmlDoc) {
    std::cout << "Error parsing document" << std::endl;
    return -1;
  }

  DOMElement* elementRoot = xmlDoc->getDocumentElement();

  xuti::readHeader(elementRoot, header);
  
  delete parser;
  cms::concurrency::xercesTerminate();

  return 0;
}

void xuti:: GetNodeStringData(xercesc::DOMNode* node, std::string& value){
  value=  cms::xerces::toString(node->getTextContent());       
}
  

