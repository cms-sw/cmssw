#include "CondTools/Ecal/interface/EcalTBWeightsXMLTranslator.h"
#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/PlatformUtils.hpp>
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

  XMLPlatformUtils::Initialize();

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

  return 0;
}

int EcalTBWeightsXMLTranslator::writeXML(const  std::string& filename,
					 const  EcalCondHeader& header,
					 const  EcalTBWeights&  record){
  std::fstream fs(filename.c_str(),ios::out);
  fs<< dumpXML(header,record);
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

  DOMElement* rowelement=0;

  // loop on row nodes
  while  (rownode){

    rowelement = dynamic_cast< xercesc::DOMElement* >(rownode);

    std::string rowid_s = toNative(rowelement->getAttribute(fromNative(id_tag).c_str()));

    std::stringstream rowid_ss(rowid_s);
    int rowid = 0;
    rowid_ss >> rowid;

    std::string weightrow = toNative(rownode->getTextContent());

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
    
    std::string rowid_s = toNative(rowelement->getAttribute( fromNative(id_tag).c_str()));
    
    std::stringstream rowid_ss(rowid_s);
    int rowid = 0;
    rowid_ss >> rowid;

    std::string weightrow = toNative(rownode->getTextContent());
    
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
    std::string rowid_s = toNative(rowelement->getAttribute(fromNative(id_tag).c_str()));

    std::stringstream rowid_ss(rowid_s);
    int rowid = 0;
    rowid_ss >> rowid;

    std::string weightrow = toNative(rownode->getTextContent());

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
    std::string rowid_s = toNative(rowelement->getAttribute(fromNative(id_tag).c_str()));

    std::stringstream rowid_ss(rowid_s);
    int rowid = 0;
    rowid_ss >> rowid;

    std::string weightrow = toNative(rownode->getTextContent());

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

  DOMElement * weightsetel= doc->createElement( fromNative(EcalWeightSet_tag).c_str());
  parentNode-> appendChild(weightsetel);

  DOMElement* wgtBS = doc->createElement( fromNative(wgtBeforeSwitch_tag).c_str());
  weightsetel->appendChild(wgtBS);
  
  DOMElement* wgtAS = doc->createElement(fromNative(wgtAfterSwitch_tag).c_str());
  weightsetel->appendChild(wgtAS);
  
  DOMElement* wgtChi2BS = doc->createElement( fromNative(wgtChi2BeforeSwitch_tag).c_str());
  weightsetel->appendChild(wgtChi2BS);
  
  DOMElement* wgtChi2AS = doc->createElement(fromNative(wgtChi2AfterSwitch_tag).c_str());
  weightsetel->appendChild(wgtChi2AS);
  
  writeWeightMatrix(wgtBS,ws.getWeightsBeforeGainSwitch());
  writeWeightMatrix(wgtAS,ws.getWeightsAfterGainSwitch());
  
  writeChi2WeightMatrix(wgtChi2BS,ws.getChi2WeightsBeforeGainSwitch());
  writeChi2WeightMatrix(wgtChi2AS,ws.getChi2WeightsBeforeGainSwitch()); 
  
}

void 
EcalTBWeightsXMLTranslator::writeWeightMatrix(xercesc::DOMNode* node,
					      const EcalWeightSet::EcalWeightMatrix& matrix){

  DOMElement* row=0;
  DOMAttr* rowid=0;
  DOMText* rowvalue=0;
 
  const int ncols =10;
  const int nrows =3;

  for(int i=0;i<nrows;++i)
    {


      row= node->getOwnerDocument()->createElement( fromNative(row_tag).c_str());
      node->appendChild(row);

      stringstream value_s;
      value_s << i; 

      rowid = node->getOwnerDocument()->createAttribute(fromNative(id_tag).c_str());
      rowid ->setValue(fromNative(value_s.str()).c_str());
      row ->setAttributeNode(rowid);

      stringstream row_s;

      for(int k=0;k<ncols;++k) row_s <<" " << matrix(i,k)<<" " ;

      rowvalue = 
	node->getOwnerDocument()->createTextNode(fromNative(row_s.str()).c_str());
      row->appendChild(rowvalue);
    }//for loop on col

  
}


void 
EcalTBWeightsXMLTranslator::writeChi2WeightMatrix(xercesc::DOMNode* node,
						  const EcalWeightSet::EcalChi2WeightMatrix& matrix){

  DOMElement* row=0;
  DOMAttr* rowid=0;
  DOMText* rowvalue=0;
 
  const int ncols =10;
  const int nrows =10;

  for(int i=0;i<nrows;++i)
    {


      row= node->getOwnerDocument()->createElement( fromNative(row_tag).c_str());
      node->appendChild(row);

      stringstream value_s;
      value_s << i; 

      rowid = node->getOwnerDocument()->createAttribute(fromNative(id_tag).c_str());
      rowid ->setValue(fromNative(value_s.str()).c_str());
      row ->setAttributeNode(rowid);

      stringstream row_s;

      for(int k=0;k<ncols;++k) row_s << " "<< matrix(i,k)<<" ";
      
      rowvalue = 
	node->getOwnerDocument()->createTextNode(fromNative(row_s.str()).c_str());
      row->appendChild(rowvalue);
    }//for loop on col

  
}


std::string EcalTBWeightsXMLTranslator::dumpXML(const EcalCondHeader& header,
						const EcalTBWeights& record){


 XMLPlatformUtils::Initialize();
  
  DOMImplementation*  impl =
    DOMImplementationRegistry::getDOMImplementation(fromNative("LS").c_str());
  
  DOMWriter* writer =static_cast<DOMImplementationLS*>(impl)->createDOMWriter( );
  writer->setFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true);
  
  DOMDocumentType* doctype = impl->createDocumentType( fromNative("XML").c_str(), 0, 0 );
  DOMDocument *    doc = 
    impl->createDocument( 0, fromNative(EcalTBWeights_tag).c_str(), doctype );
  
  
  doc->setEncoding(fromNative("UTF-8").c_str() );
  doc->setStandalone(true);
  doc->setVersion(fromNative("1.0").c_str() );
  
  
  DOMElement* root = doc->getDocumentElement();
  xuti::writeHeader(root, header);


  const EcalTBWeights::EcalTBWeightMap wmap= record.getMap();

  EcalTBWeights::EcalTBWeightMap::const_iterator it ;
  for (it =wmap.begin(); it!=wmap.end(); ++it){
   
    DOMElement * tbweight= doc->createElement(fromNative(EcalTBWeight_tag).c_str());
    root->appendChild(tbweight);

    EcalXtalGroupId          gid = it->first.first;
    EcalTBWeights::EcalTDCId tid = it->first.second;
    EcalWeightSet            ws  = it->second;

    WriteNodeWithValue(tbweight,EcalXtalGroupId_tag, gid);
    WriteNodeWithValue(tbweight,EcalTDCId_tag, tid);
    writeWeightSet(tbweight,ws);

     
  } //

  std::string dump= toNative(writer->writeToString(*root)); 
  doc->release();
  
  //   XMLPlatformUtils::Terminate();

  return dump;
}
