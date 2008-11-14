/**
 *  \file Implementation of helper functions
 *
 *  $Id: DOMHelperFunctions.cc,v 1.6 2008/11/04 15:09:58 argiro Exp $
 */


#include "CondTools/Ecal/interface/DOMHelperFunctions.h"
#include "CondTools/Ecal/interface/XMLTags.h"
#include "CondTools/Ecal/interface/XercesString.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include <xercesc/dom/DOM.hpp>
#include <xercesc/dom/DOMNode.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <sstream>

using namespace std;
using namespace xuti;
using namespace xercesc;

const DetId xuti::readCellId(xercesc::DOMNode* node){
  
 DOMNode* child = 0;

  int ieta=0;
  int iphi=0;
  int ix=0;
  int iy=0;
  int zside=0;
  
  
  for (child = node->getFirstChild(); 
       child != 0; 
       child=child->getNextSibling()){

    // handle barrel xtals 
    if ( child->getNodeType( ) == DOMNode::ELEMENT_NODE &&
	 fromNative(iEta_tag) == child->getNodeName( )) {
      
      string value_s = toNative(child->getTextContent( ));
      stringstream value_ss(value_s); 
      value_ss>> ieta;
      
    } // if 
    
    if ( child->getNodeType( ) == DOMNode::ELEMENT_NODE &&
	  fromNative(iPhi_tag) == child->getNodeName( )) {
      
      string value_s = toNative(child->getTextContent( ));
      stringstream value_ss(value_s); 
      value_ss>> iphi;
      
    } // if 
    
    if ( child->getNodeType( ) == DOMNode::ELEMENT_NODE &&
	  fromNative(iPhi_tag) == child->getNodeName( )) {
      
      string value_s = toNative( child->getTextContent( ));
      stringstream value_ss(value_s); 
      value_ss>> iphi;
      
    } // if 

    
    // handle endcap xtals
    if ( child->getNodeType( ) == DOMNode::ELEMENT_NODE &&
	  fromNative(ix_tag) == child->getNodeName( )) {
      
      string value_s = toNative( child->getTextContent( ));
      stringstream value_ss(value_s); 
      value_ss>> ix;
      
    } // if 

    if ( child->getNodeType( ) == DOMNode::ELEMENT_NODE &&
	  fromNative(iy_tag) == child->getNodeName( )) {
      
      string value_s = toNative( child->getTextContent( ));
      stringstream value_ss(value_s); 
      value_ss>> iy;
      
    } // if 
    
    if ( child->getNodeType( ) == DOMNode::ELEMENT_NODE &&
	  fromNative(zside_tag) == child->getNodeName( )) {
      
      string value_s = toNative( child->getTextContent( ));
      stringstream value_ss(value_s); 
      value_ss>> zside;
      
    } // if 
    

  } // for
  
  if (ieta && iphi)        {return EBDetId(ieta,iphi);}
  if (ix   && iy  && zside){return EEDetId(ix,iy,zside);}
  
  cerr<<"XMLCell: error reading cell, missing field ?"<<endl;
  return 0;
 
}



void  xuti::writeCellId(xercesc::DOMNode* node, 
		  const DetId& detid){
  
  
  
  if (detid.subdetId()==EcalBarrel){ 
    
    DOMElement* eta_node = 
      node->getOwnerDocument()->createElement( fromNative(iEta_tag).c_str());
    
    stringstream value_s;
    value_s <<EBDetId(detid).ieta() ;

  
    node->appendChild(eta_node);

    DOMText*   ideta = 
      node->getOwnerDocument()->createTextNode(fromNative(value_s.str()).c_str());
    eta_node->appendChild(ideta);
    

    DOMElement* phi_node = 
      node->getOwnerDocument()->createElement( fromNative(iPhi_tag).c_str());

    value_s.str("");
    value_s <<EBDetId(detid).iphi() ;

    node->appendChild(phi_node);

    DOMText*   idphi = 
      node->getOwnerDocument()->createTextNode(fromNative(value_s.str()).c_str());
    phi_node->appendChild(idphi);

  } else  if (detid.subdetId()==EcalEndcap){
    
    DOMElement* x_node = 
      node->getOwnerDocument()->createElement( fromNative(ix_tag).c_str());

    stringstream value_s;
    value_s <<EEDetId(detid).ix() ;

    x_node->setNodeValue(fromNative(value_s.str()).c_str());
    node->appendChild(x_node);

    DOMText*   idx = 
      node->getOwnerDocument()->createTextNode(fromNative(value_s.str()).c_str());
    x_node->appendChild(idx);

    DOMElement* y_node = 
      node->getOwnerDocument()->createElement( fromNative(iy_tag).c_str());

    value_s.str("");
    value_s <<EEDetId(detid).iy() ;

    y_node->setNodeValue(fromNative(value_s.str()).c_str());
    node->appendChild(y_node);

    DOMText*   idy = 
      node->getOwnerDocument()->createTextNode(fromNative(value_s.str()).c_str());
    y_node->appendChild(idy);

    DOMElement* z_node = 
      node->getOwnerDocument()->createElement(fromNative(zside_tag).c_str());

    value_s.str("");
    value_s <<EEDetId(detid).zside() ;

    z_node->setNodeValue(fromNative(value_s.str()).c_str());
    node->appendChild(z_node);
   
    DOMText*   idz = 
      node->getOwnerDocument()->createTextNode(fromNative(value_s.str()).c_str());
    z_node->appendChild(idz);
  }

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
    cout << "Error parsing document" << endl;
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
  

