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

#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"
#include "CondTools/Ecal/interface/EcalLaserAPDPNRatiosXMLTranslator.h"

#include "CondTools/Ecal/interface/DOMHelperFunctions.h"
#include "CondTools/Ecal/interface/XMLTags.h"

using namespace XERCES_CPP_NAMESPACE;
using namespace xuti;
using namespace std;


//
// TODO: write and read time map
//
//



int  EcalLaserAPDPNRatiosXMLTranslator::readXML(const std::string& filename, 
					     EcalCondHeader& header,
					     EcalLaserAPDPNRatios& record){



  XMLPlatformUtils::Initialize();

  XercesDOMParser* parser = new XercesDOMParser;
  parser->setValidationScheme( XercesDOMParser::Val_Never );
  parser->setDoNamespaces( false );
  parser->setDoSchema( false );
  
  parser->parse(filename.c_str());

  DOMDocument* xmlDoc = parser->getDocument();
  if (!xmlDoc) {
    std::cout << "EcalLaserAPDPNRatiosXMLTranslator::Error parsing document" << std::endl;
    return -1;
  }

  DOMElement* elementRoot = xmlDoc->getDocumentElement();

  xuti::readHeader(elementRoot,header);

  DOMNode * cellnode = getChildNode(elementRoot,Cell_tag);

  while(cellnode)
    {
      float p1 = 0;
      float p2 = 0;
      float p3 = 0;
//       edm::TimeStamp t1=0;
//       edm::TimeStamp t2=0;
//       edm::TimeStamp t3=0;


      DetId detid = readCellId(dynamic_cast<DOMElement*>(cellnode));

      DOMNode* p1_node = getChildNode(cellnode,Laser_p1_tag);
      GetNodeData(p1_node,p1);

      DOMNode* p2_node = getChildNode(cellnode,Laser_p2_tag);
      GetNodeData(p2_node,p2);

      DOMNode* p3_node = getChildNode(cellnode,Laser_p3_tag);
      GetNodeData(p3_node,p3);

//       DOMNode* t1_node = getChildNode(cellnode,Laser_t1_tag);
//       GetNodeData(t1_node,t1);

//       DOMNode* t2_node = getChildNode(cellnode,Laser_t2_tag);
//       GetNodeData(t3_node,t2);

//       DOMNode* p1_node = getChildNode(cellnode,Laser_t3_tag);
//       GetNodeData(t3_node,t3);

 
      EcalLaserAPDPNRatios::EcalLaserAPDPNpair pair;
      pair.p1 =p1;
      pair.p2 =p2;
      pair.p3 =p3;

      record.setValue(detid,pair);
   
      cellnode = cellnode->getNextSibling();

      while(cellnode && cellnode->getNodeType() != DOMNode::ELEMENT_NODE)
	cellnode = cellnode->getNextSibling();

 
    }  

  delete parser;
  XMLPlatformUtils::Terminate();
  return 0;
  
  
}
  




  int EcalLaserAPDPNRatiosXMLTranslator::writeXML(const std::string& filename, 
					       const EcalCondHeader& header,
					       const EcalLaserAPDPNRatios& record){
    std::fstream fs(filename.c_str(),ios::out);
    fs<< dumpXML(header,record);
    return 0;
  }


std::string EcalLaserAPDPNRatiosXMLTranslator::dumpXML(
					     const EcalCondHeader& header,
			                     const EcalLaserAPDPNRatios& record){
 

   XMLPlatformUtils::Initialize();

    DOMImplementation*  impl =
      DOMImplementationRegistry::getDOMImplementation(fromNative("LS").c_str());

    DOMWriter* writer =static_cast<DOMImplementationLS*>(impl)->createDOMWriter( );
    writer->setFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true);

    DOMDocumentType* doctype = impl->createDocumentType( fromNative("XML").c_str(), 0, 0 );
    DOMDocument *    doc = 
         impl->createDocument( 0, fromNative(Laser_tag).c_str(), doctype );


    doc->setEncoding(fromNative("UTF-8").c_str() );
    doc->setStandalone(true);
    doc->setVersion(fromNative("1.0").c_str() );

    
    DOMElement* root = doc->getDocumentElement();
 
    xuti::writeHeader(root,header);

    string Lasertag = "Laser", LMtag = "LM";
    for(int cellid = 0; cellid < (int)record.getTimeMap().size(); cellid++) {

      DOMElement* cellnode = doc->createElement( fromNative(Lasertag).c_str());
      root->appendChild(cellnode); 
      stringstream value_s;
      value_s << cellid;
      cellnode->setAttribute(fromNative(LMtag).c_str(),
			    fromNative(value_s.str()).c_str());

      long int t123[3];
      t123[0]=(record.getTimeMap())[cellid].t1.value();
      t123[1]=(record.getTimeMap())[cellid].t2.value();
      t123[2]=(record.getTimeMap())[cellid].t3.value();
      string Laser_t_tag[3] = {"t1","t2","t3"};
      for(int i = 0; i < 3; i++) {
      time_t t = t123[i] >> 32;
      char buf[256];
      struct tm lt;
      localtime_r(&t, &lt);
      strftime(buf, sizeof(buf), "%F %R:%S", &lt);
      buf[sizeof(buf)-1] = 0;
      DOMDocument* subdoc = cellnode->getOwnerDocument();
      DOMElement* new_node = subdoc->createElement(fromNative(Laser_t_tag[i]).c_str());
      cellnode->appendChild(new_node);
      std::stringstream value_ss;
      value_ss << t123[i];
      string newstr = value_ss.str() + " [" + string(buf) +"]";
      DOMText* tvalue = 
	subdoc->createTextNode(fromNative(newstr).c_str());
      new_node->appendChild(tvalue);
      }
    }
 
    for(int cellid = EBDetId::MIN_HASH;
	cellid < EBDetId::kSizeForDenseIndexing;
	++cellid)
      {

	uint32_t rawid = EBDetId::unhashIndex(cellid);

	DOMElement* cellnode= writeCell(root,rawid);	  

	float p1=(record.getLaserMap())[rawid].p1;
	float p2=(record.getLaserMap())[rawid].p2;
	float p3=(record.getLaserMap())[rawid].p3;
         
	WriteNodeWithValue(cellnode,Laser_p1_tag,p1);
	WriteNodeWithValue(cellnode,Laser_p2_tag,p2);
	WriteNodeWithValue(cellnode,Laser_p3_tag,p3);
	  	  	  
      }



   
    for(int cellid = 0;
	cellid < EEDetId::kSizeForDenseIndexing;
	++cellid)
      {
	  
	if(!EEDetId::validHashIndex(cellid)) continue;
	  
	uint32_t rawid = EEDetId::unhashIndex(cellid); 


	DOMElement* cellnode=writeCell(root,rawid);
	  
	float p1=(record.getLaserMap())[rawid].p1;
	float p2=(record.getLaserMap())[rawid].p2;
	float p3=(record.getLaserMap())[rawid].p3;
         
	WriteNodeWithValue(cellnode,Laser_p1_tag,p1);
	WriteNodeWithValue(cellnode,Laser_p2_tag,p2);
	WriteNodeWithValue(cellnode,Laser_p3_tag,p3);
      }
    

    std::string dump= toNative(writer->writeToString(*root)); 
    doc->release();
    //   XMLPlatformUtils::Terminate();

    return dump;

}
