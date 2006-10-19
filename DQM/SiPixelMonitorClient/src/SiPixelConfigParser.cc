#include "DQM/SiPixelMonitorClient/interface/SiPixelConfigParser.h"
#include "DQMServices/ClientConfig/interface/ParserFunctions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
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
  if (!doc) {
    cout << " SiPixelConfigParser::Configuration File is not set!!! " << endl;
    return false;
  }

  me_names.clear();
  unsigned int tkMapNodes = doc->getElementsByTagName(qtxml::_toDOMS("TkMap"))->getLength();
  if (tkMapNodes != 1) return false;
  /// Get Node
  DOMNode* tkMapNode = doc->getElementsByTagName(qtxml::_toDOMS("TkMap"))->item(0);
 //Get QTEST name
  if (! tkMapNode) return false;
  DOMElement* tkMapElement = static_cast<DOMElement *>(tkMapNode);          
  if (! tkMapElement) return false;		 
		
  tkmap_name = qtxml::_toString(tkMapElement->getAttribute(qtxml::_toDOMS("name")));
	
  DOMNodeList * meList 
		  = tkMapElement->getElementsByTagName(qtxml::_toDOMS("MonElement"));
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
  
}
//
// -- Read Update Frequency for the TrackerMap
//
bool SiPixelConfigParser::getFrequencyForTrackerMap(int& u_freq){
  if (!doc) {
    cout << " SiPixelConfigParser::Configuration File is not set!!! " << endl;
    return false;
  }

  unsigned int tkMapNodes = doc->getElementsByTagName(qtxml::_toDOMS("TkMap"))->getLength();
  if (tkMapNodes != 1) return false;
  /// Get Node
  DOMNode* tkMapNode = doc->getElementsByTagName(qtxml::_toDOMS("TkMap"))->item(0);
 //Get Node name
  if (! tkMapNode) return false;
  DOMElement* tkMapElement = static_cast<DOMElement *>(tkMapNode);          
  if (! tkMapElement) return false;		 
		
  u_freq = atoi(qtxml::_toString(tkMapElement->getAttribute(qtxml::_toDOMS("update_frequency"))).c_str());
  return true;
}
//
// -- Get List of MEs for the summary plot and the
//
bool SiPixelConfigParser::getMENamesForSummary(string& structure_name,
						vector<string>& me_names) {
  if (!doc) {
    cout << " SiPixelConfigParser::Configuration File is not set!!! " << endl;
    return false;
  }

  me_names.clear();
  unsigned int structureNodes = doc->getElementsByTagName(qtxml::_toDOMS("SubStructureLevel"))->getLength();
  if (structureNodes == 0) return false;
  /// Get Node
  DOMNode* structureNode = doc->getElementsByTagName(qtxml::_toDOMS("SubStructureLevel"))->item(0);
 //Get QTEST name
  if (! structureNode) return false;
  DOMElement* structureElement = static_cast<DOMElement *>(structureNode);          
  if (! structureElement) return false;		 
		
  structure_name = qtxml::_toString(structureElement->getAttribute(qtxml::_toDOMS("name"))); 

  DOMNodeList * meList 
		  = structureElement->getElementsByTagName(qtxml::_toDOMS("MonElement"));
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
  
}
//
// -- Get List of MEs for the summary plot and the
//
bool SiPixelConfigParser::getFrequencyForSummary(int& u_freq) {
  if (!doc) {
    cout << " SiPixelConfigParser::Configuration File is not set!!! " << endl;
    return false;
  }

  unsigned int structureNodes = doc->getElementsByTagName(qtxml::_toDOMS("SubStructureLevel"))->getLength();
  if (structureNodes == 0) return false;
  /// Get Node
  DOMNode* structureNode = doc->getElementsByTagName(qtxml::_toDOMS("SubStructureLevel"))->item(0);
 //Get Node name
  if (! structureNode) return false;
  DOMElement* structureElement = static_cast<DOMElement *>(structureNode);          
  if (! structureElement) return false;		 
		
  u_freq = atoi(qtxml::_toString(structureElement->getAttribute(qtxml::_toDOMS("name"))).c_str());
  return true;
}
