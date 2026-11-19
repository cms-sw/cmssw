#include "CaloOnlineTools/HcalOnlineDb/interface/XMLDOMBlock.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/XMLProcessor.h"
#include "Utilities/Xerces/interface/Xerces.h"

#include <xercesc/dom/DOM.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/util/XMLException.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/XMLUni.hpp>

#include <iostream>
#include <memory>
#include <string>

XMLProcessor* XMLProcessor::instance = nullptr;

XMLProcessor::XMLProcessor() { init(); }

XMLProcessor::~XMLProcessor() { terminate(); }

void XMLProcessor::write(XMLDOMBlock* doc, const std::string& target) {
  xercesc::DOMDocument* loader = doc->getDocument();
  serializeDOM(loader, target);
}

void XMLProcessor::serializeDOM(xercesc::DOMNode* node, const std::string& target) {
  XMLCh tempStr[100];
  xercesc::XMLString::transcode("LS", tempStr, 99);
  xercesc::DOMImplementation* impl = xercesc::DOMImplementationRegistry::getDOMImplementation(tempStr);
  xercesc::DOMLSSerializer* theSerializer = ((xercesc::DOMImplementationLS*)impl)->createLSSerializer();
  xercesc::DOMConfiguration* dc = theSerializer->getDomConfig();
  dc->setParameter(xercesc::XMLUni::fgDOMWRTDiscardDefaultContent, true);
  dc->setParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true);

  std::unique_ptr<xercesc::XMLFormatTarget> myFormTarget =
      std::make_unique<xercesc::LocalFileFormatTarget>(_toXMLCh(target).get());
  try {
    xercesc::DOMLSOutput* outputDesc = ((xercesc::DOMImplementationLS*)impl)->createLSOutput();
    outputDesc->setByteStream(myFormTarget.get());
    theSerializer->write(node, outputDesc);
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

  theSerializer->release();
}

void XMLProcessor::init(void) {
  std::cerr << "Intializing Xerces-c...";
  try {
    cms::concurrency::xercesInitialize();
  } catch (const xercesc::XMLException& toCatch) {
    std::cout << " FAILED! Exiting..." << std::endl;
    return;
  }
  std::cerr << " done" << std::endl;
}

void XMLProcessor::terminate(void) {
  std::cout << "Terminating Xerces-c...";
  cms::concurrency::xercesTerminate();
  std::cout << " done" << std::endl;
}

std::unique_ptr<XMLCh, XMLProcessor::XMLChDeleter> XMLProcessor::_toXMLCh(const int temp) {
  const std::string text = std::to_string(temp);

  return _toXMLCh(text);
}

std::unique_ptr<XMLCh, XMLProcessor::XMLChDeleter> XMLProcessor::_toXMLCh(const std::string& temp) {
  return std::unique_ptr<XMLCh, XMLChDeleter>{xercesc::XMLString::transcode(temp.c_str())};
}
