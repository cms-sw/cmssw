#include "CondTools/Ecal/interface/EcalTBWeightsXMLTranslator.h"
#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include "FWCore/Concurrency/interface/Xerces.h"
#include "Utilities/Xerces/interface/XercesStrUtils.h"
#include <xercesc/util/XMLString.hpp>
#include <xercesc/sax/SAXException.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include "CondTools/Ecal/interface/DOMHelperFunctions.h"
#include "CondTools/Ecal/interface/XMLTags.h"
#include <fstream>

using namespace xercesc;
using namespace std;
using namespace xuti;

// ouch ! I have to define here the << and >> operator !

std::ostream& operator<<( std::ostream& stream_,
			  const EcalXtalGroupId& id_ ) {
  return stream_ << id_.id();
}  


std::istream& operator>>( std::istream& stream_,
			  EcalXtalGroupId& id_ ) {
  unsigned int id;
  stream_ >> id;
  id_ =EcalXtalGroupId(id); 
  return stream_;
}  

int EcalTBWeightsXMLTranslator::readXML(const std::string& filename,
					EcalCondHeader& header,
					EcalTBWeights&  record){

  cms::concurrency::xercesInitialize();

  XercesDOMParser* parser = new XercesDOMParser;
  parser->setValidationScheme( XercesDOMParser::Val_Never );
  parser->setDoNamespaces( false );
  parser->setDoSchema( false );
  
  parser->parse(filename.c_str());

  DOMDocument* xmlDoc = parser->getDocument();
  if (!xmlDoc) {
    std::cout << "EcalTBWeightsXMLTranslator::Error parsing document" << std::endl;
    return -1;
  }

  DOMElement* elementRoot = xmlDoc->getDocumentElement();
  xuti::readHeader(elementRoot, header);

  DOMNode * wnode=getChildNode(elementRoot,EcalTBWeight_tag);

  while (wnode){

    DOMNode * gid_node = getChildNode(wnode,EcalXtalGroupId_tag);
    DOMNode * tdc_node = getChildNode(wnode,EcalTDCId_tag);
    DOMNode * ws_node =  getChildNode(wnode,EcalWeightSet_tag);
    
    EcalXtalGroupId          gid;
    EcalTBWeights::EcalTDCId tid;
    EcalWeightSet            ws;

    GetNodeData(gid_node,gid);
    GetNodeData(tdc_node,tid);
    
    readWeightSet(ws_node,ws);

    record.setValue(gid,tid,ws);

    wnode= wnode->getNextSibling();

    while (wnode&& wnode->getNodeType( ) != DOMNode::ELEMENT_NODE)      
      wnode= wnode->getNextSibling();
  }

  cms::concurrency::xercesTerminate();

  return 0;
}

int EcalTBWeightsXMLTranslator::writeXML(const  std::string& filename,
					 const  EcalCondHeader& header,
					 const  EcalTBWeights&  record){
  cms::concurrency::xercesInitialize();

  std::fstream fs(filename.c_str(),ios::out);
  fs<< dumpXML(header,record);

  cms::concurrency::xercesTerminate();

  return 0;  
}


void 
EcalTBWeightsXMLTranslator::readWeightSet(xercesc::DOMNode* parentNode, 
					  EcalWeightSet& ws){



  // get the first cell node
  DOMNode * wgtBSnode=getChildNode(parentNode,wgtBeforeSwitch_tag);
  DOMNode * wgtASnode=getChildNode(parentNode,wgtAfterSwitch_tag);
  DOMNode * wgtChi2BSnode=getChildNode(parentNode,wgtChi2BeforeSwitch_tag);
  DOMNode * wgtChi2ASnode=getChildNode(parentNode,wgtChi2AfterSwitch_tag);


  DOMNode* rownode = getChildNode(wgtBSnode,row_tag);  

  DOMElement* rowelement=nullptr;

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
    while(weightrow_s >> weight){
      ws.getWeightsBeforeGainSwitch()(rowid,i)= weight; 
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
    
    std::string rowid_s = cms::xerces::toString(rowelement->getAttribute( cms::xerces::uStr(id_tag.c_str()).ptr()));
    
    std::stringstream rowid_ss(rowid_s);
    int rowid = 0;
    rowid_ss >> rowid;

    std::string weightrow = cms::xerces::toString(rownode->getTextContent());
    
    std::stringstream weightrow_s(weightrow);
    double weight = 0;
    int i = 0;
    while(weightrow_s >> weight){
      ws.getWeightsAfterGainSwitch()(rowid,i)= weight; 
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
	ws.getChi2WeightsBeforeGainSwitch()(rowid,i)= weight; 
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
    while(weightrow_s >> weight){
      ws.getChi2WeightsAfterGainSwitch()(rowid,i)= weight; 
      i++;
    }
    
    
    // get next cell
    rownode = rownode->getNextSibling();
    
    while (rownode && rownode->getNodeType( ) != DOMNode::ELEMENT_NODE)      
      rownode = rownode->getNextSibling();
    
    
  }


}

void 
EcalTBWeightsXMLTranslator::writeWeightSet(xercesc::DOMNode* parentNode, 
					   const EcalWeightSet& ws){
  
  
  xercesc::DOMDocument* doc = parentNode->getOwnerDocument();

  DOMElement * weightsetel= doc->createElement( cms::xerces::uStr(EcalWeightSet_tag.c_str()).ptr());
  parentNode-> appendChild(weightsetel);

  DOMElement* wgtBS = doc->createElement( cms::xerces::uStr(wgtBeforeSwitch_tag.c_str()).ptr());
  weightsetel->appendChild(wgtBS);
  
  DOMElement* wgtAS = doc->createElement(cms::xerces::uStr(wgtAfterSwitch_tag.c_str()).ptr());
  weightsetel->appendChild(wgtAS);
  
  DOMElement* wgtChi2BS = doc->createElement( cms::xerces::uStr(wgtChi2BeforeSwitch_tag.c_str()).ptr());
  weightsetel->appendChild(wgtChi2BS);
  
  DOMElement* wgtChi2AS = doc->createElement(cms::xerces::uStr(wgtChi2AfterSwitch_tag.c_str()).ptr());
  weightsetel->appendChild(wgtChi2AS);
  
  writeWeightMatrix(wgtBS,ws.getWeightsBeforeGainSwitch());
  writeWeightMatrix(wgtAS,ws.getWeightsAfterGainSwitch());
  
  writeChi2WeightMatrix(wgtChi2BS,ws.getChi2WeightsBeforeGainSwitch());
  writeChi2WeightMatrix(wgtChi2AS,ws.getChi2WeightsBeforeGainSwitch()); 
  
}

void 
EcalTBWeightsXMLTranslator::writeWeightMatrix(xercesc::DOMNode* node,
					      const EcalWeightSet::EcalWeightMatrix& matrix){

  DOMElement* row=nullptr;
  DOMAttr* rowid=nullptr;
  DOMText* rowvalue=nullptr;
 
  const int ncols =10;
  const int nrows =3;

  for(int i=0;i<nrows;++i)
    {


      row= node->getOwnerDocument()->createElement( cms::xerces::uStr(row_tag.c_str()).ptr());
      node->appendChild(row);

      stringstream value_s;
      value_s << i; 

      rowid = node->getOwnerDocument()->createAttribute(cms::xerces::uStr(id_tag.c_str()).ptr());
      rowid ->setValue(cms::xerces::uStr(value_s.str().c_str()).ptr());
      row ->setAttributeNode(rowid);

      stringstream row_s;

      for(int k=0;k<ncols;++k) row_s <<" " << matrix(i,k)<<" " ;

      rowvalue = 
	node->getOwnerDocument()->createTextNode(cms::xerces::uStr(row_s.str().c_str()).ptr());
      row->appendChild(rowvalue);
    }//for loop on col

  
}


void 
EcalTBWeightsXMLTranslator::writeChi2WeightMatrix(xercesc::DOMNode* node,
						  const EcalWeightSet::EcalChi2WeightMatrix& matrix){

  DOMElement* row=nullptr;
  DOMAttr* rowid=nullptr;
  DOMText* rowvalue=nullptr;
 
  const int ncols =10;
  const int nrows =10;

  for(int i=0;i<nrows;++i)
    {


      row= node->getOwnerDocument()->createElement( cms::xerces::uStr(row_tag.c_str()).ptr());
      node->appendChild(row);

      stringstream value_s;
      value_s << i; 

      rowid = node->getOwnerDocument()->createAttribute(cms::xerces::uStr(id_tag.c_str()).ptr());
      rowid ->setValue(cms::xerces::uStr(value_s.str().c_str()).ptr());
      row ->setAttributeNode(rowid);

      stringstream row_s;

      for(int k=0;k<ncols;++k) row_s << " "<< matrix(i,k)<<" ";
      
      rowvalue = 
	node->getOwnerDocument()->createTextNode(cms::xerces::uStr(row_s.str().c_str()).ptr());
      row->appendChild(rowvalue);
    }//for loop on col

  
}


std::string EcalTBWeightsXMLTranslator::dumpXML(const EcalCondHeader& header,
						const EcalTBWeights& record){


  cms::concurrency::xercesInitialize();
  
  unique_ptr<DOMImplementation> impl( DOMImplementationRegistry::getDOMImplementation(cms::xerces::uStr("LS").ptr()));
  
  DOMLSSerializer* writer = impl->createLSSerializer();
  if( writer->getDomConfig()->canSetParameter( XMLUni::fgDOMWRTFormatPrettyPrint, true ))
    writer->getDomConfig()->setParameter( XMLUni::fgDOMWRTFormatPrettyPrint, true );
  
  DOMDocumentType* doctype = impl->createDocumentType( cms::xerces::uStr("XML").ptr(), nullptr, nullptr );
  DOMDocument* doc =
    impl->createDocument( nullptr, cms::xerces::uStr(EcalTBWeights_tag.c_str()).ptr(), doctype );
  DOMElement* root = doc->getDocumentElement();

  xuti::writeHeader(root, header);

  const EcalTBWeights::EcalTBWeightMap& wmap= record.getMap();

  EcalTBWeights::EcalTBWeightMap::const_iterator it ;
  for (it =wmap.begin(); it!=wmap.end(); ++it){
   
    DOMElement * tbweight= doc->createElement(cms::xerces::uStr(EcalTBWeight_tag.c_str()).ptr());
    root->appendChild(tbweight);

    EcalXtalGroupId          gid = it->first.first;
    EcalTBWeights::EcalTDCId tid = it->first.second;
    EcalWeightSet            ws  = it->second;

    WriteNodeWithValue(tbweight,EcalXtalGroupId_tag, gid);
    WriteNodeWithValue(tbweight,EcalTDCId_tag, tid);
    writeWeightSet(tbweight,ws);

     
  } //

  std::string dump = cms::xerces::toString( writer->writeToString( root ));
  doc->release();
  doctype->release();
  writer->release();

  return dump;
}
