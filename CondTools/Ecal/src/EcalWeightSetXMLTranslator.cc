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

#include "CondTools/Ecal/interface/EcalWeightSetXMLTranslator.h"
#include "CondTools/Ecal/interface/XMLTags.h"
#include "CondTools/Ecal/interface/DOMHelperFunctions.h"

using namespace XERCES_CPP_NAMESPACE;
using namespace xuti;
using namespace std;

int  EcalWeightSetXMLTranslator::readXML(const std::string& filename, 
					 EcalCondHeader& header,
					 EcalWeightSet& record){

  cms::concurrency::xercesInitialize();

  XercesDOMParser* parser = new XercesDOMParser;
  parser->setValidationScheme( XercesDOMParser::Val_Never );
  parser->setDoNamespaces( false );
  parser->setDoSchema( false );
  
  parser->parse(filename.c_str());

  DOMDocument* xmlDoc = parser->getDocument();
  if (!xmlDoc) {
    std::cout << "EcalWeightSetXMLTranslator::Error parsing document" << std::endl;
    return -1;
  }

  DOMElement* elementRoot = xmlDoc->getDocumentElement();

  xuti::readHeader(elementRoot, header);
  // get the first cell node
  DOMNode * wgtBSnode=getChildNode(elementRoot,wgtBeforeSwitch_tag);
  DOMNode * wgtASnode=getChildNode(elementRoot,wgtAfterSwitch_tag);
  DOMNode * wgtChi2BSnode=getChildNode(elementRoot,wgtChi2BeforeSwitch_tag);
  DOMNode * wgtChi2ASnode=getChildNode(elementRoot,wgtChi2AfterSwitch_tag);

  DOMNode* rownode = getChildNode(wgtBSnode,row_tag);  

  DOMElement* rowelement;

  // loop on row nodes
  while  (rownode){

    rowelement = dynamic_cast< xercesc::DOMElement* >(rownode);

    std::string rowid_s = cms::xerces::toString(rowelement->getAttribute(cms::xerces::uStr(id_tag.c_str()).ptr()));

    std::stringstream rowid_ss(rowid_s);
    int rowid = 0;
    rowid_ss >> rowid;

    std::string weightrow = cms::xerces::toString(rownode->getTextContent());

    std::stringstream weightrow_s(weightrow);
    double weight = 0;
    int i = 0;
    while(weightrow_s >> weight)
      {
	record.getWeightsBeforeGainSwitch()(i,rowid)= weight; 
	i++;
      }

    
    // get next cell
    rownode = rownode->getNextSibling();
    
    while (rownode && rownode->getNodeType( ) != DOMNode::ELEMENT_NODE)      
      rownode = rownode->getNextSibling();
  }

  rownode = getChildNode(wgtASnode,row_tag);  

  // loop on row nodes
  while  (rownode){

    rowelement = dynamic_cast< xercesc::DOMElement* >(rownode);

    std::string rowid_s = cms::xerces::toString(rowelement->getAttribute(cms::xerces::uStr(id_tag.c_str()).ptr()));

    std::stringstream rowid_ss(rowid_s);
    int rowid = 0;
    rowid_ss >> rowid;

    std::string weightrow = cms::xerces::toString(rownode->getTextContent());

    std::stringstream weightrow_s(weightrow);
    double weight = 0;
    int i = 0;
    while(weightrow_s >> weight)
      {
	record.getWeightsAfterGainSwitch()(i,rowid)= weight; 
	i++;
      }
    
    
    // get next cell
    rownode = rownode->getNextSibling();
    
    while (rownode && rownode->getNodeType( ) != DOMNode::ELEMENT_NODE)      
      rownode = rownode->getNextSibling();
    
    
  }
  
  rownode = getChildNode(wgtChi2BSnode,row_tag);  


  // loop on row nodes
  while  (rownode){

    rowelement = dynamic_cast< xercesc::DOMElement* >(rownode);
    std::string rowid_s = cms::xerces::toString(rowelement->getAttribute(cms::xerces::uStr(id_tag.c_str()).ptr()));

    std::stringstream rowid_ss(rowid_s);
    int rowid = 0;
    rowid_ss >> rowid;

    std::string weightrow = cms::xerces::toString(rownode->getTextContent());

    std::stringstream weightrow_s(weightrow);
    double weight = 0;
    int i = 0;
    while(weightrow_s >> weight)
      {
	record.getChi2WeightsBeforeGainSwitch()(i,rowid)= weight; 
	i++;
      }
    
    
    // get next cell
    rownode = rownode->getNextSibling();
    
    while (rownode && rownode->getNodeType( ) != DOMNode::ELEMENT_NODE)      
      rownode = rownode->getNextSibling();
    
    
  }


  rownode = getChildNode(wgtChi2ASnode,row_tag);  


  // loop on row nodes
  while  (rownode){

    rowelement = dynamic_cast< xercesc::DOMElement* >(rownode);  
    std::string rowid_s = cms::xerces::toString(rowelement->getAttribute(cms::xerces::uStr(id_tag.c_str()).ptr()));

    std::stringstream rowid_ss(rowid_s);
    int rowid = 0;
    rowid_ss >> rowid;

    std::string weightrow = cms::xerces::toString(rownode->getTextContent());

    std::stringstream weightrow_s(weightrow);
    double weight = 0;
    int i = 0;
    while(weightrow_s >> weight)
      {
	record.getChi2WeightsAfterGainSwitch()(i,rowid)= weight; 
	i++;
      }
    
    
    // get next cell
    rownode = rownode->getNextSibling();
    
    while (rownode && rownode->getNodeType( ) != DOMNode::ELEMENT_NODE)      
      rownode = rownode->getNextSibling();
    
    
  }

  delete parser;
  cms::concurrency::xercesTerminate();
  return 0;    
}

int EcalWeightSetXMLTranslator::writeXML(const std::string& filename,
					 const EcalCondHeader& header, 
					 const EcalWeightSet& record){
    
  cms::concurrency::xercesInitialize();

  std::fstream fs(filename.c_str(),ios::out);
  fs<< dumpXML(header,record);

  cms::concurrency::xercesTerminate();

  return 0;
}

void EcalWeightSetXMLTranslator::write10x10(xercesc::DOMElement* node,
					    const EcalWeightSet& record){

  DOMElement* row[10];
  DOMAttr* rowid[10];
  DOMText* rowvalue[10];
  EcalWeightSet::EcalChi2WeightMatrix echi2wmatrix;


  if(cms::xerces::toString(node->getNodeName()) == wgtChi2BeforeSwitch_tag)
    {
      echi2wmatrix = record.getChi2WeightsBeforeGainSwitch();

    }
  if(cms::xerces::toString(node->getNodeName()) == wgtChi2AfterSwitch_tag)
    {
      echi2wmatrix = record.getChi2WeightsAfterGainSwitch();

    }

  for(int i=0;i<10;++i)
    {


      row[i] = node->getOwnerDocument()->createElement(cms::xerces::uStr(row_tag.c_str()).ptr());
      node->appendChild(row[i]);

      stringstream value_s;
      value_s << i; 

      rowid[i] = node->getOwnerDocument()->createAttribute(cms::xerces::uStr(id_tag.c_str()).ptr());
      rowid[i]->setValue(cms::xerces::uStr(value_s.str().c_str()).ptr());
      row[i]->setAttributeNode(rowid[i]);

      stringstream row_s;

      for(int k=0;k<10;++k)
	{
	  row_s << " ";
	  row_s << echi2wmatrix(k,i);
	  row_s << " ";
	}//for loop on element


      rowvalue[i] = node->getOwnerDocument()->createTextNode(cms::xerces::uStr(row_s.str().c_str()).ptr());
      row[i]->appendChild(rowvalue[i]);
    }//for loop on row
}



void EcalWeightSetXMLTranslator::write3x10(xercesc::DOMElement* node,
					   const EcalWeightSet& record){

  DOMElement* row[10];
  DOMAttr* rowid[10];
  DOMText* rowvalue[10];
  EcalWeightSet::EcalWeightMatrix ewmatrix;


  if(cms::xerces::toString(node->getNodeName()) == wgtBeforeSwitch_tag)
    ewmatrix = record.getWeightsBeforeGainSwitch();
  
  if(cms::xerces::toString(node->getNodeName()) == wgtAfterSwitch_tag)
    ewmatrix = record.getWeightsAfterGainSwitch();
  
  
  for(int i=0;i<10;++i)
    {

      row[i] = node->getOwnerDocument()->createElement(cms::xerces::uStr(row_tag.c_str()).ptr());
      node->appendChild(row[i]);

      stringstream value_s;
      value_s << i; 

      rowid[i] = node->getOwnerDocument()->createAttribute(cms::xerces::uStr(id_tag.c_str()).ptr());

      rowid[i]->setValue(cms::xerces::uStr(value_s.str().c_str()).ptr());

      row[i]->setAttributeNode(rowid[i]);


      stringstream row_s;

      for(int k=0;k<3;++k)
	{
	  row_s << " ";
	  row_s << ewmatrix(k,i);
	  row_s << " ";
	}//for loop on element


      rowvalue[i] = node->getOwnerDocument()->createTextNode(cms::xerces::uStr(row_s.str().c_str()).ptr());
      row[i]->appendChild(rowvalue[i]);
    }//for loop on row
}


std::string EcalWeightSetXMLTranslator::dumpXML(const EcalCondHeader& header,
						const EcalWeightSet&  record){

  unique_ptr<DOMImplementation> impl( DOMImplementationRegistry::getDOMImplementation(cms::xerces::uStr("LS").ptr()));
  
  DOMLSSerializer* writer = impl->createLSSerializer();
  if( writer->getDomConfig()->canSetParameter( XMLUni::fgDOMWRTFormatPrettyPrint, true ))
    writer->getDomConfig()->setParameter( XMLUni::fgDOMWRTFormatPrettyPrint, true );
  
  DOMDocumentType* doctype = impl->createDocumentType( cms::xerces::uStr("XML").ptr(), nullptr, nullptr );
  DOMDocument *    doc = 
    impl->createDocument( nullptr,cms::xerces::uStr(EcalWeightSet_tag.c_str()).ptr(), doctype );
  
  DOMElement* root = doc->getDocumentElement();

  xuti::writeHeader(root, header);

  DOMElement* wgtBS = doc->createElement(cms::xerces::uStr(wgtBeforeSwitch_tag.c_str()).ptr());
  root->appendChild(wgtBS);
  
  DOMElement* wgtAS = doc->createElement(cms::xerces::uStr(wgtAfterSwitch_tag.c_str()).ptr());
  root->appendChild(wgtAS);
  
  DOMElement* wgtChi2BS = doc->createElement(cms::xerces::uStr(wgtChi2BeforeSwitch_tag.c_str()).ptr());
  root->appendChild(wgtChi2BS);
  
  DOMElement* wgtChi2AS = doc->createElement(cms::xerces::uStr(wgtChi2AfterSwitch_tag.c_str()).ptr());
  root->appendChild(wgtChi2AS);
  
  write3x10(wgtBS,record);
  write3x10(wgtAS,record);
  
  write10x10(wgtChi2BS,record);
  write10x10(wgtChi2AS,record);
  
  std::string dump = cms::xerces::toString(writer->writeToString( root )); 
  doc->release();
  doctype->release();
  writer->release();
  
  return dump;
}
