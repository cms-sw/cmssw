#include "DQM/SiPixelMonitorClient/interface/SiPixelConfigParser.h"
#include "DQM/SiPixelMonitorClient/interface/ANSIColors.h"
#include "DQMServices/ClientConfig/interface/ParserFunctions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace xercesc;
using namespace std;

//
// -- Constructor
// 
SiPixelConfigParser::SiPixelConfigParser() : DQMParserBase() {
  edm::LogInfo("SiPixelConfigParser") << 
    " Creating SiPixelConfigParser " << "\n" ;
}
//
// --  Destructor
// 
SiPixelConfigParser::~SiPixelConfigParser() {
  edm::LogInfo("SiPixelActionExecutor") << 
    " Deleting SiPixelConfigParser " << "\n" ;
}
//
// -- Read ME list for the TrackerMap
//
bool SiPixelConfigParser::getMENamesForTrackerMap(string& tkmap_name, 
						 vector<string>& me_names){
  if (!doc()) {
    cout << " SiPixelConfigParser::Configuration File is not set!!! " << endl;
    return false;
  }

  me_names.clear();
  unsigned int tkMapNodes = doc()->getElementsByTagName(qtxml::_toDOMS("TkMap"))->getLength();
  if (tkMapNodes != 1) 
  {
   cout << ACYellow << ACBold 
        << "[SiPixelConfigParser::getMENamesForTrackerMap()]"
	<< ACRed << ACBold
	<< "No TkMap tag found in configuration file"
	<< ACPlain << endl ;
   return false;
  }
  /// Get Node
  DOMNode* tkMapNode = doc()->getElementsByTagName(qtxml::_toDOMS("TkMap"))->item(0);
 //Get QTEST name
  if (! tkMapNode) 
  {
   cout << ACYellow << ACBold 
        << "[SiPixelConfigParser::getMENamesForTrackerMap()]"
	<< ACRed << ACBold
	<< " No TkMap tag elements found in configuration file"
	<< ACPlain << endl ;
   return false;
  }
  DOMElement* tkMapElement = static_cast<DOMElement *>(tkMapNode);          
  if (! tkMapElement) 
  {
   cout << ACYellow << ACBold 
        << "[SiPixelConfigParser::getMENamesForTrackerMap()]"
	<< ACRed << ACBold
	<< " No TkMap tag dom elements found in configuration file"
	<< ACPlain << endl ;
   return false;		 
  }		
  tkmap_name = qtxml::_toString(tkMapElement->getAttribute(qtxml::_toDOMS("name")));
	
  DOMNodeList * meList 
		  = tkMapElement->getElementsByTagName(qtxml::_toDOMS("MonElement"));
  if( meList->getLength() == 0 )
  {
    cout << ACYellow << ACBold 
    	 << "[SiPixelConfigParser::getMENamesForTrackerMap()]"
    	 << ACRed << ACBold
    	 << " No MonElement found in configuration file"
    	 << ACPlain << endl ;
  }
  for (unsigned int k = 0; k < meList->getLength(); k++) {
    DOMNode* meNode = meList->item(k);
    if (!meNode) 
    {
     cout << ACYellow << ACBold 
     	  << "[SiPixelConfigParser::getMENamesForTrackerMap()]"
     	  << ACRed << ACBold
     	  << " No MonElement item found in configuration file"
     	  << ACPlain << endl ;
     return false;
    }
    DOMElement* meElement = static_cast<DOMElement *>(meNode);          
    if (!meElement) 
    { 
     cout << ACYellow << ACBold 
     	  << "[SiPixelConfigParser::getMENamesForTrackerMap()]"
     	  << ACRed << ACBold
     	  << " No MonElement sub-elements found in configuration file"
     	  << ACPlain << endl ;
     return false;
    }
    string me_name = qtxml::_toString(meElement->getAttribute (qtxml::_toDOMS ("name"))); 
    me_names.push_back(me_name);    
  }
  if (me_names.size() == 0) 
  {
    cout << ACYellow << ACBold 
    	 << "[SiPixelConfigParser::getMENamesForTrackerMap()]"
    	 << ACRed << ACBold
    	 << " No MonElement sub-element names found in configuration file"
    	 << ACPlain << endl ;
   return false;
  } else {
   return true;
  }
}
//
// -- Read Update Frequency for the TrackerMap
//
bool SiPixelConfigParser::getFrequencyForTrackerMap(int& u_freq){
  if (!doc()) {
    cout << " SiPixelConfigParser::Configuration File is not set!!! " << endl;
    return false;
  }

  unsigned int tkMapNodes = doc()->getElementsByTagName(qtxml::_toDOMS("TkMap"))->getLength();
  if (tkMapNodes != 1) return false;
  /// Get Node
  DOMNode* tkMapNode = doc()->getElementsByTagName(qtxml::_toDOMS("TkMap"))->item(0);
 //Get Node name
  if (! tkMapNode) return false;
  DOMElement* tkMapElement = static_cast<DOMElement *>(tkMapNode);          
  if (! tkMapElement) return false;		 
		
  u_freq = atoi(qtxml::_toString(tkMapElement->getAttribute(qtxml::_toDOMS("update_frequency"))).c_str());
  return true;
}
//
// -- Get List of MEs for the module tree plots:
//
bool SiPixelConfigParser::getMENamesForTree(string& structure_name,
						vector<string>& me_names) {
  //cout<<"Entering SiPixelConfigParser::getMENamesForTree..."<<endl;
  if (!doc()) {
    cout << " SiPixelConfigParser::Configuration File is not set!!! " << endl;
    return false;
  }

  me_names.clear();
  unsigned int structureNodes = doc()->getElementsByTagName(qtxml::_toDOMS("SubStructureLevel"))->getLength();
  if (structureNodes == 0) return false;
  /// Get Node
  DOMNode* structureNode = doc()->getElementsByTagName(qtxml::_toDOMS("SubStructureLevel"))->item(0);
 //Get QTEST name
  if (! structureNode) return false;
  DOMElement* structureElement = static_cast<DOMElement *>(structureNode);          
  if (! structureElement) return false;		 
		
  structure_name = qtxml::_toString(structureElement->getAttribute(qtxml::_toDOMS("name"))); 

  DOMNodeList * meList = structureElement->getElementsByTagName(qtxml::_toDOMS("MonElement"));
  for (unsigned int k = 0; k < meList->getLength(); k++) {
    DOMNode* meNode = meList->item(k);
    if (!meNode) return false;
    DOMElement* meElement = static_cast<DOMElement *>(meNode);          
    if (!meElement) return false;
    string me_name = qtxml::_toString(meElement->getAttribute (qtxml::_toDOMS ("name"))); 
    me_names.push_back(me_name);    
  }
  if (me_names.size() == 0) return false;
  else return true;
  //cout<<"...leaving SiPixelConfigParser::getMENamesForTree!"<<endl;
  
}
//
// -- Get List of MEs for the summary plot and the
//
bool SiPixelConfigParser::getMENamesForBarrelSummary(string& structure_name,
						vector<string>& me_names) {
//  cout<<"Entering SiPixelConfigParser::getMENamesForBarrelSummary..."<<endl;
  if (!doc()) {
    cout << " SiPixelConfigParser::Configuration File is not set!!! " << endl;
    return false;
  }
  me_names.clear();
  unsigned int structureNodes = doc()->getElementsByTagName(qtxml::_toDOMS("SubStructureBarrelLevel"))->getLength();
  if (structureNodes == 0) return false;
  /// Get Node
  DOMNode* structureNode = doc()->getElementsByTagName(qtxml::_toDOMS("SubStructureBarrelLevel"))->item(0);
 //Get QTEST name
  if (! structureNode) return false;
  DOMElement* structureElement = static_cast<DOMElement *>(structureNode);          
  if (! structureElement) return false;		 
  structure_name = qtxml::_toString(structureElement->getAttribute(qtxml::_toDOMS("name"))); 

  DOMNodeList * meList = structureElement->getElementsByTagName(qtxml::_toDOMS("MonElement"));
  for (unsigned int k = 0; k < meList->getLength(); k++) {
    DOMNode* meNode = meList->item(k);
    if (!meNode) return false;
    DOMElement* meElement = static_cast<DOMElement *>(meNode);          
    if (!meElement) return false;
    string me_name = qtxml::_toString(meElement->getAttribute (qtxml::_toDOMS ("name"))); 
    me_names.push_back(me_name);    
  }
//  cout<<"...leaving SiPixelConfigParser::getMENamesForBarrelSummary!"<<endl;
  if (me_names.size() == 0) return false;
  else return true;
  
}
bool SiPixelConfigParser::getMENamesForEndcapSummary(string& structure_name,
						vector<string>& me_names) {
//  cout<<"Entering SiPixelConfigParser::getMENamesForEndcapSummary..."<<endl;
  if (!doc()) {
    cout << " SiPixelConfigParser::Configuration File is not set!!! " << endl;
    return false;
  }

  me_names.clear();
  unsigned int structureNodes = doc()->getElementsByTagName(qtxml::_toDOMS("SubStructureEndcapLevel"))->getLength();
  if (structureNodes == 0) return false;
  /// Get Node
  DOMNode* structureNode = doc()->getElementsByTagName(qtxml::_toDOMS("SubStructureEndcapLevel"))->item(0);
 //Get QTEST name
  if (! structureNode) return false;
  DOMElement* structureElement = static_cast<DOMElement *>(structureNode);          
  if (! structureElement) return false;		 
		
  structure_name = qtxml::_toString(structureElement->getAttribute(qtxml::_toDOMS("name"))); 

  DOMNodeList * meList = structureElement->getElementsByTagName(qtxml::_toDOMS("MonElement"));
  for (unsigned int k = 0; k < meList->getLength(); k++) {
    DOMNode* meNode = meList->item(k);
    if (!meNode) return false;
    DOMElement* meElement = static_cast<DOMElement *>(meNode);          
    if (!meElement) return false;
    string me_name = qtxml::_toString(meElement->getAttribute (qtxml::_toDOMS ("name"))); 
    me_names.push_back(me_name);    
  }
  if (me_names.size() == 0) return false;
  else return true;
//  cout<<"...leaving SiPixelConfigParser::getMENamesForEndcapSummary!"<<endl;
  
}


bool SiPixelConfigParser::getMENamesForFEDErrorSummary(string& structure_name,
						       vector<string>& me_names) {
  //cout<<"Entering SiPixelConfigParser::getMENamesForFEDErrorSummary..."<<endl;
  if (!doc()) {
    cout << " SiPixelConfigParser::Configuration File is not set!!! " << endl;
    return false;
  }

  me_names.clear();
  unsigned int structureNodes = doc()->getElementsByTagName(qtxml::_toDOMS("SubStructureNonDetId"))->getLength();
  if (structureNodes == 0) return false;
  /// Get Node
  DOMNode* structureNode = doc()->getElementsByTagName(qtxml::_toDOMS("SubStructureNonDetId"))->item(0);
 //Get QTEST name
  if (! structureNode) return false;
  DOMElement* structureElement = static_cast<DOMElement *>(structureNode);          
  if (! structureElement) return false;		 
		
  structure_name = qtxml::_toString(structureElement->getAttribute(qtxml::_toDOMS("name"))); 

  DOMNodeList * meList = structureElement->getElementsByTagName(qtxml::_toDOMS("MonElement"));
  for (unsigned int k = 0; k < meList->getLength(); k++) {
    DOMNode* meNode = meList->item(k);
    if (!meNode) return false;
    DOMElement* meElement = static_cast<DOMElement *>(meNode);          
    if (!meElement) return false;
    string me_name = qtxml::_toString(meElement->getAttribute (qtxml::_toDOMS ("name"))); 
    me_names.push_back(me_name);    
  }
  if (me_names.size() == 0) return false;
  else return true;
  //cout<<"...leaving SiPixelConfigParser::getMENamesForFEDErrorSummary!"<<endl;
  
}
////
// -- Get List of MEs for the summary plot and the
//
bool SiPixelConfigParser::getFrequencyForBarrelSummary(int& u_freq) {
  if (!doc()) {
    cout << " SiPixelConfigParser::Configuration File is not set!!! " << endl;
    return false;
  }

  unsigned int structureNodes = doc()->getElementsByTagName(qtxml::_toDOMS("SubStructureBarrelLevel"))->getLength();
  if (structureNodes == 0) return false;
  /// Get Node
  DOMNode* structureNode = doc()->getElementsByTagName(qtxml::_toDOMS("SubStructureBarrelLevel"))->item(0);
 //Get Node name
  if (! structureNode) return false;
  DOMElement* structureElement = static_cast<DOMElement *>(structureNode);          
  if (! structureElement) return false;		 
		
  u_freq = atoi(qtxml::_toString(structureElement->getAttribute(qtxml::_toDOMS("update_frequency"))).c_str());
  return true;
}


bool SiPixelConfigParser::getFrequencyForEndcapSummary(int& u_freq) {
  if (!doc()) {
    cout << " SiPixelConfigParser::Configuration File is not set!!! " << endl;
    return false;
  }

  unsigned int structureNodes = doc()->getElementsByTagName(qtxml::_toDOMS("SubStructureEndcapLevel"))->getLength();
  if (structureNodes == 0) return false;
  /// Get Node
  DOMNode* structureNode = doc()->getElementsByTagName(qtxml::_toDOMS("SubStructureEndcapLevel"))->item(0);
 //Get Node name
  if (! structureNode) return false;
  DOMElement* structureElement = static_cast<DOMElement *>(structureNode);          
  if (! structureElement) return false;		 
		
  u_freq = atoi(qtxml::_toString(structureElement->getAttribute(qtxml::_toDOMS("update_frequency"))).c_str());
  return true;
}


bool SiPixelConfigParser::getMENamesForGrandBarrelSummary(string& structure_name,
						vector<string>& me_names) {
  //cout<<"Entering SiPixelConfigParser::getMENamesForGrandBarrelSummary..."<<endl;
  if (!doc()) {
    cout << " SiPixelConfigParser::Configuration File is not set!!! " << endl;
    return false;
  }

  me_names.clear();
  unsigned int structureNodes = doc()->getElementsByTagName(qtxml::_toDOMS("SubStructureGrandBarrelLevel"))->getLength();
  if (structureNodes == 0) return false;
  /// Get Node
  DOMNode* structureNode = doc()->getElementsByTagName(qtxml::_toDOMS("SubStructureGrandBarrelLevel"))->item(0);
 //Get QTEST name
  if (! structureNode) return false;
  DOMElement* structureElement = static_cast<DOMElement *>(structureNode);          
  if (! structureElement) return false;		 
		
  structure_name = qtxml::_toString(structureElement->getAttribute(qtxml::_toDOMS("name"))); 

  DOMNodeList * meList = structureElement->getElementsByTagName(qtxml::_toDOMS("MonElement"));
  for (unsigned int k = 0; k < meList->getLength(); k++) {
    DOMNode* meNode = meList->item(k);
    if (!meNode) return false;
    DOMElement* meElement = static_cast<DOMElement *>(meNode);          
    if (!meElement) return false;
    string me_name = qtxml::_toString(meElement->getAttribute (qtxml::_toDOMS ("name"))); 
    me_names.push_back(me_name);    
  }
  if (me_names.size() == 0) return false;
  else return true;
  //cout<<"...leaving SiPixelConfigParser::getMENamesForGrandBarrelSummary!"<<endl;
  
}


bool SiPixelConfigParser::getMENamesForGrandEndcapSummary(string& structure_name,
						vector<string>& me_names) {
  //cout<<"Entering SiPixelConfigParser::getMENamesForGrandEndcapSummary..."<<endl;
  if (!doc()) {
    cout << " SiPixelConfigParser::Configuration File is not set!!! " << endl;
    return false;
  }

  me_names.clear();
  unsigned int structureNodes = doc()->getElementsByTagName(qtxml::_toDOMS("SubStructureGrandEndcapLevel"))->getLength();
  if (structureNodes == 0) return false;
  /// Get Node
  DOMNode* structureNode = doc()->getElementsByTagName(qtxml::_toDOMS("SubStructureGrandEndcapLevel"))->item(0);
 //Get QTEST name
  if (! structureNode) return false;
  DOMElement* structureElement = static_cast<DOMElement *>(structureNode);          
  if (! structureElement) return false;		 
		
  structure_name = qtxml::_toString(structureElement->getAttribute(qtxml::_toDOMS("name"))); 

  DOMNodeList * meList = structureElement->getElementsByTagName(qtxml::_toDOMS("MonElement"));
  for (unsigned int k = 0; k < meList->getLength(); k++) {
    DOMNode* meNode = meList->item(k);
    if (!meNode) return false;
    DOMElement* meElement = static_cast<DOMElement *>(meNode);          
    if (!meElement) return false;
    string me_name = qtxml::_toString(meElement->getAttribute (qtxml::_toDOMS ("name"))); 
    me_names.push_back(me_name);    
  }
  if (me_names.size() == 0) return false;
  else return true;
  //cout<<"...leaving SiPixelConfigParser::getMENamesForGrandEndcapSummary!"<<endl;
  
}


bool SiPixelConfigParser::getFrequencyForGrandBarrelSummary(int& u_freq) {
  if (!doc()) {
    cout << " SiPixelConfigParser::Configuration File is not set!!! " << endl;
    return false;
  }

  unsigned int structureNodes = doc()->getElementsByTagName(qtxml::_toDOMS("SubStructureGrandBarrelLevel"))->getLength();
  if (structureNodes == 0) return false;
  /// Get Node
  DOMNode* structureNode = doc()->getElementsByTagName(qtxml::_toDOMS("SubStructureGrandBarrelLevel"))->item(0);
 //Get Node name
  if (! structureNode) return false;
  DOMElement* structureElement = static_cast<DOMElement *>(structureNode);          
  if (! structureElement) return false;		 
		
  u_freq = atoi(qtxml::_toString(structureElement->getAttribute(qtxml::_toDOMS("update_frequency"))).c_str());
  return true;
}


bool SiPixelConfigParser::getFrequencyForGrandEndcapSummary(int& u_freq) {
  if (!doc()) {
    cout << " SiPixelConfigParser::Configuration File is not set!!! " << endl;
    return false;
  }

  unsigned int structureNodes = doc()->getElementsByTagName(qtxml::_toDOMS("SubStructureGrandEndcapLevel"))->getLength();
  if (structureNodes == 0) return false;
  /// Get Node
  DOMNode* structureNode = doc()->getElementsByTagName(qtxml::_toDOMS("SubStructureGrandEndcapLevel"))->item(0);
 //Get Node name
  if (! structureNode) return false;
  DOMElement* structureElement = static_cast<DOMElement *>(structureNode);          
  if (! structureElement) return false;		 
		
  u_freq = atoi(qtxml::_toString(structureElement->getAttribute(qtxml::_toDOMS("update_frequency"))).c_str());
  return true;
}


bool SiPixelConfigParser::getMessageLimitForQTests(int& u_freq) {
  if (!doc()) {
    cout << " SiPixelConfigParser::Configuration File is not set!!! " << endl;
    return false;
  }
  unsigned int structureNodes = doc()->getElementsByTagName(qtxml::_toDOMS("QTestMessageLimit"))->getLength();
  if (structureNodes == 0) return false;
  /// Get Node
  DOMNode* structureNode = doc()->getElementsByTagName(qtxml::_toDOMS("QTestMessageLimit"))->item(0);
 //Get Node name
  if (! structureNode) return false;
  DOMElement* structureElement = static_cast<DOMElement *>(structureNode);          
  if (! structureElement) return false;
		
  u_freq = atoi(qtxml::_toString(structureElement->getAttribute(qtxml::_toDOMS("value"))).c_str());
  return true;
}



bool SiPixelConfigParser::getSourceType(int& u_freq) {
  if (!doc()) {
    cout << " SiPixelConfigParser::Configuration File is not set!!! " << endl;
    return false;
  }
  unsigned int structureNodes = doc()->getElementsByTagName(qtxml::_toDOMS("SourceType"))->getLength();
  if (structureNodes == 0) return false;
  /// Get Node
  DOMNode* structureNode = doc()->getElementsByTagName(qtxml::_toDOMS("SourceType"))->item(0);
 //Get Node name
  if (! structureNode) return false;
  DOMElement* structureElement = static_cast<DOMElement *>(structureNode);          
  if (! structureElement) return false;
		
  u_freq = atoi(qtxml::_toString(structureElement->getAttribute(qtxml::_toDOMS("code"))).c_str());
  return true;
}

bool SiPixelConfigParser::getCalibType(int& u_freq) {
  if (!doc()) {
    cout << " SiPixelConfigParser::Configuration File is not set!!! " << endl;
    return false;
  }
  unsigned int structureNodes = doc()->getElementsByTagName(qtxml::_toDOMS("CalibType"))->getLength();
  if (structureNodes == 0) return false;
  /// Get Node
  DOMNode* structureNode = doc()->getElementsByTagName(qtxml::_toDOMS("CalibType"))->item(0);
 //Get Node name
  if (! structureNode) return false;
  DOMElement* structureElement = static_cast<DOMElement *>(structureNode);          
  if (! structureElement) return false;
		
  u_freq = atoi(qtxml::_toString(structureElement->getAttribute(qtxml::_toDOMS("value"))).c_str());
  return true;
}




