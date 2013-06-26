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

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondTools/Ecal/interface/EcalAlignmentXMLTranslator.h"
#include "CondTools/Ecal/interface/DOMHelperFunctions.h"

using namespace XERCES_CPP_NAMESPACE;
using namespace xuti;
using namespace std;

int EcalAlignmentXMLTranslator::writeXML(const string& filename, 
					 const EcalCondHeader& header,
					 const Alignments& record){
  fstream fs(filename.c_str(),ios::out);
  fs<< dumpXML(header,record);
  return 0;
}

string EcalAlignmentXMLTranslator::dumpXML(const EcalCondHeader& header,
					  const Alignments& record){

  XMLPlatformUtils::Initialize();
  
  DOMImplementation*  impl =
    DOMImplementationRegistry::getDOMImplementation(fromNative("LS").c_str());
  
  DOMWriter* writer =static_cast<DOMImplementationLS*>(impl)->createDOMWriter( );
  writer->setFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true);
  
  DOMDocumentType* doctype = impl->createDocumentType(fromNative("XML").c_str(), 0, 0 );
  DOMDocument *    doc = 
    impl->createDocument( 0, fromNative(AlignmentConstant_tag).c_str(), doctype );


  doc->setEncoding(fromNative("UTF-8").c_str() );
  doc->setStandalone(true);
  doc->setVersion(fromNative("1.0").c_str() );
    
  DOMElement* root = doc->getDocumentElement();
 
  xuti::writeHeader(root,header);

  for ( vector<AlignTransform>::const_iterator it = record.m_align.begin();
	it != record.m_align.end(); it++ ) {
    int Id = (*it).rawId();
    int sub = (Id>>24)&0xF;
    stringstream subdet;
    if(sub == 2) {
      subdet << "EB";
      int SM ;
      int phi = Id&0x1FF;
      //    int eta = (Id>>9)&0x3F;
      int side = (Id>>16)&1;
      if(side == 0) {
	subdet << "-";
	SM = (phi + 19)/20;
      }
      else{
	subdet << "+";
	SM = phi/20;
      }
      if(SM < 10) subdet << "0" << SM;
      else subdet << SM;
    }
    else if(sub == 4) {
      subdet << "EE";
      //      int y = Id&0x7F;
      int x = (Id>>7)&0x7F;
      //      int Dee = x/70 + 1;
      int side = (Id>>14)&1;
      if(side == 0) {
	subdet << "-";
      }
      else{
	subdet << "+";
      }
      //      subdet << "0" << Dee;
      if(x == 20) subdet << "F";
      else if(x == 70) subdet << "N";
      else cout << " strange value for x " << x << endl; // should never occur
    }
    else if(sub == 6) {
      subdet << "ES";
      //      int strip = Id&0x3F;
      int x = (Id>>6)&0x3F;
      //      int y = (Id>>12)&0x3F;
      int plane = (Id>>18)&1;
      int side = (Id>>19)&1;
      if(side == 0) subdet << "-";
      else subdet << "+";
      if(plane) subdet << "F";
      else subdet << "R";
      if(x/30) subdet << "F";
      else subdet << "N";
    }
    else cout << " problem sub = " << sub << endl;
    cout << (*it).rawId()
	      << " " << (*it).rotation().getPhi()
	      << " " << (*it).rotation().getTheta() 
	      << " " << (*it).rotation().getPsi() 
	      << " " << (*it).translation().x()
	      << " " << (*it).translation().y()
	      << " " << (*it).translation().z()
	      << endl;
    uint32_t rawid = (*it).rawId();
    DOMElement* cellnode = 
      root->getOwnerDocument()->createElement( fromNative(Cell_tag).c_str());
    root->appendChild(cellnode);  
    cellnode->setAttribute(fromNative(subdet_tag).c_str(),
    			   fromNative(subdet.str()).c_str());
    xuti::WriteNodeWithValue(cellnode, id_tag, rawid);
    xuti::WriteNodeWithValue(cellnode, x_tag, (*it).translation().x());
    xuti::WriteNodeWithValue(cellnode, y_tag, (*it).translation().y());
    xuti::WriteNodeWithValue(cellnode, z_tag, (*it).translation().z());
    xuti::WriteNodeWithValue(cellnode, Phi_tag, (*it).rotation().getPhi());
    xuti::WriteNodeWithValue(cellnode, Theta_tag, (*it).rotation().getTheta());
    xuti::WriteNodeWithValue(cellnode, Psi_tag, (*it).rotation().getPsi());
  }

  string dump = toNative(writer->writeToString(*root)); 
  doc->release();
  return dump;
}

