#include "DQMServices/ClientConfig/interface/DQMParserBase.h"
#include "DQMServices/ClientConfig/interface/ParserFunctions.h"


#include <stdexcept>         
/** \file
 *
 *  Implementation of DQMParserBase
 *
 *  $Date: 2011/06/16 03:07:28 $
 *  $Revision: 1.8 $
 *  \author Ilaria Segoni
 */


using namespace xercesc;

DQMParserBase::DQMParserBase(){
	parser=0; 
}

DQMParserBase::~DQMParserBase(){
	delete parser;
	parser=0; 
}


void DQMParserBase::getDocument(std::string configFile, bool UseDB){
	
    parser = new XercesDOMParser;     
    parser->setValidationScheme(XercesDOMParser::Val_Auto);
    parser->setDoNamespaces(false);
    if(UseDB){
//       std::cout<<"=== This is config file from getDocument ====== "<<std::endl;
//       std::cout<<configFile<<std::endl;
      MemBufInputSource mb((const XMLByte*)configFile.c_str(),strlen(configFile.c_str()),"",false);
      parser->parse(mb);
    }
    else{
      parser->parse(configFile.c_str()); 
    }
    xercesc::DOMDocument* doc = parser->getDocument();
    assert(doc);

}

void DQMParserBase::getNewDocument(std::string configFile, bool UseDB){
  parser->resetDocumentPool();
  if(UseDB){
    std::cout<<"=== This is config file from getNewDocument ==== "<<std::endl;
    std::cout<<configFile<<std::endl;
    MemBufInputSource mb((const XMLByte*)configFile.c_str(),strlen(configFile.c_str()),"",false);
    parser->parse(mb);
  }
  else{
    parser->parse(configFile.c_str()); 
  }
  xercesc::DOMDocument* doc = parser->getDocument();
  assert(doc);

}
int DQMParserBase::countNodes(std::string tagName){
	unsigned int tagsNum  = 
	  parser->getDocument()->getElementsByTagName(qtxml::_toDOMS(tagName))->getLength();
	return tagsNum;
}
