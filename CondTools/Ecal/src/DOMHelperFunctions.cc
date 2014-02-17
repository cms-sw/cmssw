/**
 *  \file Implementation of helper functions
 *
 *  $Id: DOMHelperFunctions.cc,v 1.6 2011/01/21 12:48:59 fwyzard Exp $
 */


#include "CondTools/Ecal/interface/DOMHelperFunctions.h"
#include "CondTools/Ecal/interface/XMLTags.h"
#include "CondTools/Ecal/interface/XercesString.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include <xercesc/dom/DOM.hpp>
#include <xercesc/dom/DOMNode.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/PlatformUtils.hpp>
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

 
  ieta_str << toNative(node->getAttribute(fromNative(iEta_tag).c_str()));
  iphi_str << toNative(node->getAttribute(fromNative(iPhi_tag).c_str()));
  ix_str   << toNative(node->getAttribute(fromNative(ix_tag).c_str()));
  iy_str   << toNative(node->getAttribute(fromNative(iy_tag).c_str()));
  ixSC_str << toNative(node->getAttribute(fromNative(ixSC_tag).c_str()));
  iySC_str << toNative(node->getAttribute(fromNative(iySC_tag).c_str()));
  zside_str<< toNative(node->getAttribute(fromNative(zside_tag).c_str()));
  
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
      node->getOwnerDocument()->createElement( fromNative(Cell_tag).c_str());

  node->appendChild(cell_node);
  
  if (detid.subdetId() == EcalBarrel ){ 
   
    stringstream value_s;
    value_s <<EBDetId(detid).ieta() ;

    cell_node->setAttribute(fromNative(iEta_tag).c_str(),
			    fromNative(value_s.str()).c_str());
    value_s.str("");
    value_s <<EBDetId(detid).iphi() ;

    cell_node->setAttribute(fromNative(iPhi_tag).c_str(),
			    fromNative(value_s.str()).c_str());

  } else if (detid.subdetId() == EcalEndcap){
    
    // is it a EcalScDetId ?
    unsigned int ScIdCheck = detid.rawId() & 0x00008000;
    if (ScIdCheck == 0) {
      stringstream value_s;
      value_s <<EEDetId(detid).ix() ;
  
      cell_node->setAttribute(fromNative(ix_tag).c_str(),
			      fromNative(value_s.str()).c_str());
      value_s.str("");
      value_s <<EEDetId(detid).iy() ;

      cell_node->setAttribute(fromNative(iy_tag).c_str(),
			      fromNative(value_s.str()).c_str());
      value_s.str("");
      value_s <<EEDetId(detid).zside() ;

      cell_node->setAttribute(fromNative(zside_tag).c_str(),
			      fromNative(value_s.str()).c_str());
    }
    else {
      stringstream value_s;
      value_s << EcalScDetId(detid).ix() ;
  
      cell_node->setAttribute(fromNative(ixSC_tag).c_str(),
			      fromNative(value_s.str()).c_str());
      value_s.str("");
      value_s << EcalScDetId(detid).iy() ;

      cell_node->setAttribute(fromNative(iySC_tag).c_str(),
			      fromNative(value_s.str()).c_str());
      value_s.str("");
      value_s << EcalScDetId(detid).zside() ;

      cell_node->setAttribute(fromNative(zside_tag).c_str(),
			      fromNative(value_s.str()).c_str());
    }

  } else if (detid.subdetId() == EcalTriggerTower ){ 
    stringstream value_s;
    value_s <<EcalTrigTowerDetId(detid).ieta() ;
  
    cell_node->setAttribute(fromNative(iEta_tag).c_str(),
			    fromNative(value_s.str()).c_str());
    value_s.str("");
    value_s <<EcalTrigTowerDetId(detid).iphi() ;

    cell_node->setAttribute(fromNative(iPhi_tag).c_str(),
			    fromNative(value_s.str()).c_str());
  }
  return cell_node;
}
  
// return 0 if not found
DOMNode * xuti::getChildNode(DOMNode * node,  const std::string& nodename ){

   if (!node) return 0;

  for (DOMNode* childNode = node->getFirstChild(); 
       childNode; childNode = childNode->getNextSibling()) {
    
    if (childNode->getNodeType() == DOMNode::ELEMENT_NODE) {
      
      const string foundName = toNative(childNode->getNodeName());
      
      if (foundName == nodename) return childNode;
    }// if element 
  }// for child 
  
  return 0;

}



void xuti::writeHeader (xercesc::DOMNode* parentNode, 
			const EcalCondHeader& header ){


  
  DOMElement* headernode =
    parentNode->getOwnerDocument()->createElement( fromNative(Header_tag).c_str());
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
  
  XMLPlatformUtils::Initialize();

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
  XMLPlatformUtils::Terminate();

  return 0;
}

void xuti:: GetNodeStringData(xercesc::DOMNode* node, std::string& value){
  value=  toNative(node->getTextContent());       
}
  

