/*  =====================================================================================
 *
 *       Filename:  CSCDQM_Collection.cc
 *
 *    Description:  Histogram booking code
 *
 *        Version:  1.0
 *        Created:  04/18/2008 03:39:49 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), Valdas.Rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 *  =====================================================================================
 */

#include "DQM/CSCMonitorModule/interface/CSCDQM_Collection.h"

namespace cscdqm {

  using namespace XERCES_CPP_NAMESPACE;
  
  Collection::Collection(const std::string p_bookingFile) {
    bookingFile = p_bookingFile;
    load();
  }
  
  /**
   * @brief  Load XML file and create definitions
   * @param  
   * @return 
   */
  void Collection::load() {

    XMLPlatformUtils::Initialize();
    XercesDOMParser *parser = new XercesDOMParser();
    parser->setValidationScheme(XercesDOMParser::Val_Always);
    parser->setDoNamespaces(true);
    parser->setDoSchema(true);
    parser->setValidationSchemaFullChecking(false); // this is default
    parser->setCreateEntityReferenceNodes(true);  // this is default
    parser->setIncludeIgnorableWhitespace (false);

    parser->parse(bookingFile.c_str());
    DOMDocument *doc = parser->getDocument();
    DOMNode *docNode = (DOMNode*) doc->getDocumentElement();

    DOMNodeList *itemList = docNode->getChildNodes();

    for(uint32_t i=0; i < itemList->getLength(); i++) {

      std::string nodeName = XMLString::transcode(itemList->item(i)->getNodeName());
      if(nodeName != "Histogram") {
        continue;
      }

      DOMNodeList *props  = itemList->item(i)->getChildNodes();
      CoHistoProps hp;
      std::string prefix = "", name = "";
      for(uint32_t j = 0; j < props->getLength(); j++) {
        std::string tname  = XMLString::transcode(props->item(j)->getNodeName());
        std::string tvalue = XMLString::transcode(props->item(j)->getTextContent());
        hp.insert(std::make_pair(tname, tvalue));
        if(tname == "Name")   name   = tvalue;
        if(tname == "Prefix") prefix = tvalue;
      }

      if(!name.empty() && !prefix.empty()) {
        CoHistoMap::iterator it = collection.find(prefix);
        if( it == collection.end()) {
          CoHisto h;
          h.insert(make_pair(name, hp));
          collection.insert(make_pair(prefix, h)); 
        } else {
          it->second.insert(make_pair(name, hp));
        }
      }

    }

    delete parser;

  }
  
}
