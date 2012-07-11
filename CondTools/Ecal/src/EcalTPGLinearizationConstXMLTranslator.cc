#include <iostream>
#include <sstream>
#include <fstream>

#include "CondFormats/EcalObjects/interface/EcalTPGLinearizationConst.h"
#include "CondTools/Ecal/interface/EcalTPGLinearizationConstXMLTranslator.h"

#include "CondTools/Ecal/interface/DOMHelperFunctions.h"
#include "CondTools/Ecal/interface/XMLTags.h"

using namespace XERCES_CPP_NAMESPACE;
using namespace xuti;
using namespace std;

int EcalTPGLinearizationConstXMLTranslator::writeXML(const std::string& filename, 
						     const EcalCondHeader& header,
						     const EcalTPGLinearizationConst& record){
  std::fstream fs(filename.c_str(),ios::out);
  fs<< dumpXML(header,record);
  return 0;
}

std::string EcalTPGLinearizationConstXMLTranslator::dumpXML(const EcalCondHeader& header,
							    const EcalTPGLinearizationConst& record){

  XMLPlatformUtils::Initialize();

  DOMImplementation*  impl =
    DOMImplementationRegistry::getDOMImplementation(fromNative("LS").c_str());

  DOMWriter* writer =static_cast<DOMImplementationLS*>(impl)->createDOMWriter( );
  writer->setFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true);

  DOMDocumentType* doctype = impl->createDocumentType( fromNative("XML").c_str(), 0, 0 );
  DOMDocument *    doc = 
    impl->createDocument( 0, fromNative(Linearization_tag).c_str(), doctype );


  doc->setEncoding(fromNative("UTF-8").c_str() );
  doc->setStandalone(true);
  doc->setVersion(fromNative("1.0").c_str() );

  DOMElement* root = doc->getDocumentElement();

  xuti::writeHeader(root,header);

  // open also a flat text file
  std::ofstream fout;
  fout.open("Linearization.txt");

  for(int cellid = EBDetId::MIN_HASH;
      cellid < EBDetId::kSizeForDenseIndexing;
      ++cellid) {

    uint32_t rawid = EBDetId::unhashIndex(cellid);

    DOMElement* cellnode= writeCell(root,rawid);	  

    float m12=(record)[rawid].mult_x12;
    float s12=(record)[rawid].shift_x12;
    float m6 =(record)[rawid].mult_x6;
    float s6 =(record)[rawid].shift_x6;
    float m1 =(record)[rawid].mult_x1;
    float s1 =(record)[rawid].shift_x1;

    WriteNodeWithValue(cellnode,Linearization_m12_tag,m12);
    WriteNodeWithValue(cellnode,Linearization_m6_tag,m6);
    WriteNodeWithValue(cellnode,Linearization_m1_tag,m1);	  	  	  
    WriteNodeWithValue(cellnode,Linearization_s12_tag,s12);
    WriteNodeWithValue(cellnode,Linearization_s6_tag,s6);
    WriteNodeWithValue(cellnode,Linearization_s1_tag,s1);
    fout << rawid << " " << m12 << " " << m6 << " " << m1 << " " << s12 << " " << s6 << " " << s1 << "\n";	  	  	  
  }

  for(int cellid = 0;
      cellid < EEDetId::kSizeForDenseIndexing;
      ++cellid) {
    if(!EEDetId::validHashIndex(cellid)) continue;
 
    uint32_t rawid = EEDetId::unhashIndex(cellid); 

    DOMElement* cellnode=writeCell(root,rawid);
  
    float m12=(record)[rawid].mult_x12;
    float s12=(record)[rawid].shift_x12;
    float m6 =(record)[rawid].mult_x6;
    float s6 =(record)[rawid].shift_x6;
    float m1 =(record)[rawid].mult_x1;
    float s1 =(record)[rawid].shift_x1;
     
    WriteNodeWithValue(cellnode,Linearization_m12_tag,m12);
    WriteNodeWithValue(cellnode,Linearization_m6_tag,m6);
    WriteNodeWithValue(cellnode,Linearization_m1_tag,m1);	  	  	  
    WriteNodeWithValue(cellnode,Linearization_s12_tag,s12);
    WriteNodeWithValue(cellnode,Linearization_s6_tag,s6);
    WriteNodeWithValue(cellnode,Linearization_s1_tag,s1);	  	  	  
    fout << rawid << " " << m12 << " " << m6 << " " << m1 << " " << s12 << " " << s6 << " " << s1 << "\n";	  	  	  
  
  }  
  fout.close();
  std::string dump= toNative(writer->writeToString(*root)); 
  doc->release();

  return dump;
}
