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

  Collection::Collection() {
  }

  
  /**
   * @brief  Load XML file and create definitions
   * @param  bookingFile Booking file to load
   * @return 
   */
  void Collection::load(const std::string bookingFile) {

    LOG_INFO << "Booking histograms from " << bookingFile;

    try {

      XMLPlatformUtils::Initialize();

      boost::shared_ptr<XercesDOMParser> parser(new XercesDOMParser());

      parser->setValidationScheme(XercesDOMParser::Val_Always);
      parser->setDoNamespaces(true);
      parser->setDoSchema(true);
      parser->setExitOnFirstFatalError(true);
      parser->setValidationConstraintFatal(true);
      BookingFileErrorHandler eh;
      parser->setErrorHandler(&eh);

      parser->parse(bookingFile.c_str());
      DOMDocument *doc = parser->getDocument();
      DOMNode *docNode = (DOMNode*) doc->getDocumentElement();

      DOMNodeList *itemList = docNode->getChildNodes();

      /// Load histogram definitions
      CoHisto definitions;
      for(uint32_t i = 0; i < itemList->getLength(); i++) {

        DOMNode* node = itemList->item(i);
        if (!isNodeElement(node) || !isNodeName(node, XML_BOOK_DEFINITION)) { continue; }

        CoHistoProps dp;
        getNodeProperties(node, dp);

        DOMElement* el = dynamic_cast<DOMElement*>(node);
        std::string id(XMLString::transcode(el->getAttribute(XMLString::transcode(XML_BOOK_DEFINITION_ID))));
        definitions.insert(make_pair(id, dp));

      }

      /// Loading histograms
      for(uint32_t i = 0; i < itemList->getLength(); i++) {

        DOMNode* node = itemList->item(i);
        if (!isNodeElement(node) || !isNodeName(node, XML_BOOK_HISTOGRAM)) { continue; }

        CoHistoProps hp;

        DOMElement* el = dynamic_cast<DOMElement*>(node);
        if (el->hasAttribute(XMLString::transcode(XML_BOOK_DEFINITION_REF))) {
          std::string id(XMLString::transcode(el->getAttribute(XMLString::transcode(XML_BOOK_DEFINITION_REF))));
          CoHistoProps d = definitions[id];
          for (CoHistoProps::iterator it = d.begin(); it != d.end(); it++) {
            hp[it->first] = it->second;
          }
        }

        getNodeProperties(node, hp);

        std::string name   = hp["Name"];
        std::string prefix = hp["Prefix"];

        CoHistoMap::iterator it = collection.find(prefix);
        if (it == collection.end()) {
          CoHisto h;
          h[name] = hp;
          collection[prefix] = h; 
        } else {
          it->second.insert(make_pair(name, hp));
        }

      }

    } catch (XMLException& e) {
      char* message = XMLString::transcode( e.getMessage() );
      throw Exception(message);
    }

  }
  
  void Collection::getNodeProperties(DOMNode*& node, CoHistoProps& p) const {
    DOMNodeList *props  = node->getChildNodes();
    for(uint32_t j = 0; j < props->getLength(); j++) {
      DOMNode* node = props->item(j);
      if(isNodeElement(node)) { continue; }
      std::string name  = XMLString::transcode(node->getNodeName());
      std::string value = XMLString::transcode(node->getTextContent());
      p[name] = value;
    }
  }

  const bool Collection::isNodeElement(DOMNode*& node) const {
    return (node->getNodeType() == DOMNode::ELEMENT_NODE);
  }

  const bool Collection::isNodeName(DOMNode*& node, const std::string name) const {
    std::string nodeName = XMLString::transcode(node->getNodeName());
    return (nodeName.compare(name) == 0);
  }

}
