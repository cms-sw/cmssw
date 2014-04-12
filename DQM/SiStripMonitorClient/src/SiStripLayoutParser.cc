#include "DQM/SiStripMonitorClient/interface/SiStripLayoutParser.h"
#include "DQMServices/ClientConfig/interface/ParserFunctions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

//
// -- Constructor
// 
SiStripLayoutParser::SiStripLayoutParser() : DQMParserBase() {
  edm::LogInfo("SiStripLayoutParser") << 
    " Creating SiStripLayoutParser " << "\n" ;
}
//
// --  Destructor
// 
SiStripLayoutParser::~SiStripLayoutParser() {
  edm::LogInfo("SiStripActionExecutor") << 
    " Deleting SiStripLayoutParser " << "\n" ;
}
//
// -- Get list of Layouts for ME groups
//
bool SiStripLayoutParser::getAllLayouts(std::map<std::string, std::vector< std::string > >& layouts){
  if (!doc()) {
    std::cout << " SiStripLayoutParser::Configuration File is not set!!! " << std::endl;
    return false;
  }

  layouts.clear();

  xercesc::DOMNodeList * layoutList 
    = doc()->getElementsByTagName(qtxml::_toDOMS("layout"));

  unsigned int nlayout = layoutList->getLength();
  if (nlayout == 0) return false;

  for (unsigned int k = 0; k < layoutList->getLength(); k++) {
    xercesc::DOMNode* layoutNode = layoutList->item(k);
    if (!layoutNode) return false;
    
    xercesc::DOMElement* layoutElement = static_cast<xercesc::DOMElement *>(layoutNode);          
    if (!layoutElement) return false;
    std::string layoutName = qtxml::_toString(layoutElement->getAttribute (qtxml::_toDOMS ("name"))); 

    xercesc::DOMNodeList * meList 
		  = layoutElement->getElementsByTagName(qtxml::_toDOMS("monitorable"));
    std::vector<std::string> me_names;
    for (unsigned int l = 0; l < meList->getLength(); l++) {
      xercesc::DOMNode* meNode = meList->item(l);
      if (!meNode) return false;
      xercesc::DOMElement* meElement = static_cast<xercesc::DOMElement *>(meNode);          
      if (!meElement) return false;
      std::string meName = qtxml::_toString(meElement->getAttribute (qtxml::_toDOMS ("name"))); 
      me_names.push_back(meName);    
    }
    if (me_names.size() > 0) layouts[layoutName] = me_names;
  }
  if ( layouts.size() > 0) return true;
  else return false; 
}
