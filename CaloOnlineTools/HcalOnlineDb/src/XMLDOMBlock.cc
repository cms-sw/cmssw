#include "CaloOnlineTools/HcalOnlineDb/interface/XMLDOMBlock.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLProcessor.h"

#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLException.hpp>
#include <xercesc/util/XMLString.hpp>

#include <iostream>
#include <memory>
#include <string>

XMLDOMBlock& XMLDOMBlock::operator+=(const XMLDOMBlock& other) {
  xercesc::DOMNodeList* _children = other.getDocumentConst()->getDocumentElement()->getChildNodes();
  int _length = _children->getLength();
  xercesc::DOMNode* _node;
  for (int i = 0; i != _length; i++) {
    _node = _children->item(i);
    xercesc::DOMNode* i_node = this->getDocument()->importNode(_node, true);
    this->getDocument()->getDocumentElement()->appendChild(i_node);
  }

  return *this;
}

XMLDOMBlock::XMLDOMBlock(const std::string& xmlFileName, bool fromScratch) {
  theProcessor = XMLProcessor::getInstance();

  if (fromScratch) {
    xercesc::DOMImplementation* impl = xercesc::DOMImplementation::getImplementation();

    document = impl->createDocument(nullptr, XMLProcessor::_toXMLCh(xmlFileName).get(), nullptr);
  } else {
    parser = std::make_unique<xercesc::XercesDOMParser>();
    parser->setValidationScheme(xercesc::XercesDOMParser::Val_Always);
    parser->setDoNamespaces(true);

    errHandler = std::make_unique<xercesc::HandlerBase>();
    parser->setErrorHandler(errHandler.get());

    try {
      parser->parse(xmlFileName.c_str());
    } catch (const xercesc::XMLException& toCatch) {
      char* message = xercesc::XMLString::transcode(toCatch.getMessage());
      std::cout << "Exception message is: \n" << message << "\n";
      xercesc::XMLString::release(&message);
    } catch (const xercesc::DOMException& toCatch) {
      char* message = xercesc::XMLString::transcode(toCatch.msg);
      std::cout << "Exception message is: \n" << message << "\n";
      xercesc::XMLString::release(&message);
    } catch (...) {
      std::cout << "Unexpected Exception \n";
    }

    document = parser->getDocument();
  }
}

xercesc::DOMDocument* XMLDOMBlock::getDocument(void) { return document; }

const xercesc::DOMDocument* XMLDOMBlock::getDocumentConst(void) const { return document; }

void XMLDOMBlock::write(const std::string& target) { theProcessor->write(this, target); }

XMLDOMBlock::~XMLDOMBlock() = default;
