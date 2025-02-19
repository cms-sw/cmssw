#include "DQM/SiPixelMonitorClient/interface/SiPixelLayoutParser.h"
#include "DQMServices/ClientConfig/interface/ParserFunctions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace xercesc;
using namespace std;

//
// -- Constructor
// 
SiPixelLayoutParser::SiPixelLayoutParser() : DQMParserBase() {
  edm::LogInfo("SiPixelLayoutParser") << 
    " Creating SiPixelLayoutParser " << "\n" ;
  cout << " Creating SiPixelLayoutParser " << endl;
}
//
// --  Destructor
// 
SiPixelLayoutParser::~SiPixelLayoutParser() {
  edm::LogInfo("SiPixelActionExecutor") << 
    " Deleting SiPixelLayoutParser " << "\n" ;
}
//
// -- Get list of Layouts for ME groups
//
bool SiPixelLayoutParser::getAllLayouts(map<string, vector< string > >& layouts){
  if (!doc()) {
    cout << " SiPixelLayoutParser::Configuration File is not set!!! " << endl;
    return false;
  }

  layouts.clear();

  DOMNodeList * layoutList 
    = doc()->getElementsByTagName(qtxml::_toDOMS("layout"));

  unsigned int nlayout = layoutList->getLength();
  if (nlayout == 0) return false;

  for (unsigned int k = 0; k < layoutList->getLength(); k++) {
    DOMNode* layoutNode 
      = layoutList->item(k);
    if (!layoutNode) return false;
    
    DOMElement* layoutElement = static_cast<DOMElement *>(layoutNode);          
    if (!layoutElement) return false;
    string layoutName = qtxml::_toString(layoutElement->getAttribute (qtxml::_toDOMS ("name"))); 

    DOMNodeList * meList 
      = layoutElement->getElementsByTagName(qtxml::_toDOMS("monitorable"));
    vector<string> me_names;
    for (unsigned int l = 0; l < meList->getLength(); l++) {
      DOMNode* meNode = meList->item(l);
      if (!meNode) return false;
      DOMElement* meElement = static_cast<DOMElement *>(meNode);          
      if (!meElement) return false;
      string meName = qtxml::_toString(meElement->getAttribute (qtxml::_toDOMS ("name"))); 
      me_names.push_back(meName);    
    }
    if (me_names.size() > 0) layouts[layoutName] = me_names;
  }
  if ( layouts.size() > 0) return true;
  else return false; 
}
