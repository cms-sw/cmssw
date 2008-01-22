#include "DQM/SiPixelMonitorClient/interface/SiPixelQTestsParser.h"
#include "DQMServices/ClientConfig/interface/ParserFunctions.h"
#include "DQMServices/ClientConfig/interface/QTestNames.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace xercesc;
using namespace std;

//
// -- Constructor
// 
SiPixelQTestsParser::SiPixelQTestsParser() : DQMParserBase() {

  testActivationOFF_ = "false";
  testON_            = "true";

  edm::LogInfo("SiPixelQTestsParser") << 
    " Creating SiPixelQTestsParser " << "\n" ;
  cout << " Creating SiPixelQTestsParser " << endl;
}
//
// --  Destructor
// 
SiPixelQTestsParser::~SiPixelQTestsParser() {

  edm::LogInfo("SiPixelActionExecutor") << 
    " Deleting SiPixelQTestsParser " << "\n" ;
}


//
// -- Get list of QTests for ME group
//
bool SiPixelQTestsParser::getAllQTests(map<string, map<string, string> >& mapQTests){
  if (!doc) {
    cout << " SiPixelQTestsParser::Configuration File is not set!!! " << endl;
    return true;
  }

  mapQTests.clear();

  DOMNodeList * qtestList = 
    doc->getElementsByTagName(qtxml::_toDOMS("QTEST"));

  unsigned int nqtest = qtestList->getLength();
  if (nqtest == 0) return true;

  for (unsigned int k = 0; k < qtestList->getLength(); ++k) {
    /// Get Node
    DOMNode* qtestNode = 
      qtestList->item(k);
    if (!qtestNode) return true;
    
    ///Get QTEST name
    DOMElement* qtestElement = 
      static_cast<DOMElement *>(qtestNode);          
    if (!qtestElement) return true;

    string qtestName = 
      qtxml::_toString(qtestElement->getAttribute (qtxml::_toDOMS ("name"))); 

    string activate = 
      qtxml::_toString (qtestElement->getAttribute (qtxml::_toDOMS ("activate"))); 
    if(!strcmp(activate.c_str(),testActivationOFF_.c_str())) return true;
    else{
      ///Get Qtest TYPE
      DOMNodeList *typeNodePrefix = 
	qtestElement->getElementsByTagName (qtxml::_toDOMS ("TYPE"));
      
      if (typeNodePrefix->getLength () != 1) return true;
      
      
      DOMElement *prefixNode = 
	dynamic_cast <DOMElement *> (typeNodePrefix->item (0));
      if (!prefixNode) return true;
      
      DOMText *prefixText = 
	dynamic_cast <DOMText *> (prefixNode->getFirstChild());
      if (!prefixText) return true;			
      
      string qtestType = 
	qtxml::_toString (prefixText->getData ());
      
      mapQTests[qtestName] = 
	this->getParams(qtestElement, qtestType);
    }
    
  } //loop on qtestTagsNum
  return false;
}


map<string, string> SiPixelQTestsParser::getParams(DOMElement* qtestElement, 
						   string qtestType){
	
  map<string, string> paramNamesValues;
  paramNamesValues[dqm::qtest_config::type] = qtestType;
  
  DOMNodeList *arguments = 
    qtestElement->getElementsByTagName (qtxml::_toDOMS ("PARAM"));
  
  for (unsigned int i=0; i<arguments->getLength(); ++i){
    DOMElement *argNode = 
      dynamic_cast <DOMElement *> ( arguments ->item(i));
    string regExp = 
      qtxml::_toString (argNode->getAttribute (qtxml::_toDOMS ("name"))); 

    DOMText *argText = 
      dynamic_cast <DOMText *> (argNode->getFirstChild());
    if (!argText){
      break;
    }
    string regExpValue = 
      qtxml::_toString (argText->getData());
    
    paramNamesValues[regExp]=regExpValue;
  }
  
  return paramNamesValues;
  
}


bool SiPixelQTestsParser::monitorElementTestsMap(std::map<std::string, std::vector<std::string> >& mapMEsQTests){
	
  unsigned int linkTagsNum  = 
    doc->getElementsByTagName(qtxml::_toDOMS("LINK"))->getLength();
  
  for (unsigned int i=0; i<linkTagsNum; ++i){
    
    DOMNode* linkNode = 
      doc->getElementsByTagName(qtxml::_toDOMS("LINK"))->item(i);
    ///Get ME name
    if (! linkNode) return true;

    DOMElement* linkElement = 
      static_cast<DOMElement *>(linkNode);          
    if (! linkElement) return true;		 

    string linkName = 
      qtxml::_toString (linkElement->getAttribute (qtxml::_toDOMS ("name"))); 
    
    DOMNodeList *testList = 
      linkElement->getElementsByTagName (qtxml::_toDOMS ("TestName"));
    unsigned int numberOfTests = 
      testList->getLength();
    
    vector<string> qualityTestList;
    for(unsigned int tt=0; tt<numberOfTests; ++tt){

      DOMElement * testElement = 
	dynamic_cast <DOMElement *> ( testList ->item(tt));		
      if (!testElement ) return true;
      
      string activate = 
	qtxml::_toString (testElement ->getAttribute (qtxml::_toDOMS ("activate"))); 
      
      DOMText *argText = 
	dynamic_cast <DOMText *> (testElement ->getFirstChild());
      
      if(!strcmp(activate.c_str(),testON_.c_str()))  {				
	if (!argText) return true;
	else{
	  string regExpValue = 
	    qtxml::_toString (argText->getData());
	  // Create List of QTests to unattach from current ME
	  qualityTestList.push_back(regExpValue);		
	}		
      }				
    }
    
    if(qualityTestList.size()) mapMEsQTests[linkName] = qualityTestList;

  }///Loop on linkTagsNum
  
  return false;
}


