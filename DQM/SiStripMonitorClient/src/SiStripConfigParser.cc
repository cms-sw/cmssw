#include "DQM/SiStripMonitorClient/interface/SiStripConfigParser.h"
#include "DQMServices/ClientConfig/interface/ParserFunctions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

//
// -- Constructor
// 
SiStripConfigParser::SiStripConfigParser() {
  edm::LogInfo("SiStripConfigParser") << 
    " Creating SiStripConfigParser " << "\n" ;
}
//
// --  Destructor
// 
SiStripConfigParser::~SiStripConfigParser() {
  edm::LogInfo("SiStripActionExecutor") << 
    " Deleting SiStripConfigParser " << "\n" ;
}
//
// -- Read ME list for the TrackerMap
//
bool SiStripConfigParser::getMENamesForTrackerMap(string& tkmap_name,
						 vector<string>& me_names){
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
// -- Get List of MEs for the summary plot and the
//
bool SiStripConfigParser::getMENamesForSummary(string& structure_name,
						vector<string>& me_names) {
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
