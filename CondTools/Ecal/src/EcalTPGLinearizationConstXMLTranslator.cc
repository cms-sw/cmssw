#include <iostream>
#include <sstream>
#include <fstream>

#include "CondFormats/EcalObjects/interface/EcalTPGLinearizationConst.h"
#include "CondTools/Ecal/interface/EcalTPGLinearizationConstXMLTranslator.h"

#include "CondTools/Ecal/interface/DOMHelperFunctions.h"
#include "CondTools/Ecal/interface/XMLTags.h"
#include "FWCore/Concurrency/interface/Xerces.h"
#include "Utilities/Xerces/interface/XercesStrUtils.h"

using namespace XERCES_CPP_NAMESPACE;
using namespace xuti;
using namespace std;

int EcalTPGLinearizationConstXMLTranslator::writeXML(const std::string& filename, 
						     const EcalCondHeader& header,
						     const EcalTPGLinearizationConst& record){
  cms::concurrency::xercesInitialize();

  std::fstream fs(filename.c_str(),ios::out);
  fs<< dumpXML(header,record);

  cms::concurrency::xercesTerminate();

  return 0;
}

std::string EcalTPGLinearizationConstXMLTranslator::dumpXML(const EcalCondHeader& header,
							    const EcalTPGLinearizationConst& record){

  unique_ptr<DOMImplementation> impl( DOMImplementationRegistry::getDOMImplementation(cms::xerces::uStr("LS").ptr()));
  
  DOMLSSerializer* writer = impl->createLSSerializer();
  if( writer->getDomConfig()->canSetParameter( XMLUni::fgDOMWRTFormatPrettyPrint, true ))
    writer->getDomConfig()->setParameter( XMLUni::fgDOMWRTFormatPrettyPrint, true );
  
  DOMDocumentType* doctype = impl->createDocumentType( cms::xerces::uStr("XML").ptr(), nullptr, nullptr );
  DOMDocument* doc =
    impl->createDocument( nullptr, cms::xerces::uStr(Linearization_tag.c_str()).ptr(), doctype );
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
  std::string dump = cms::xerces::toString( writer->writeToString( root ));
  doc->release();
  doctype->release();
  writer->release();

  return dump;
}
