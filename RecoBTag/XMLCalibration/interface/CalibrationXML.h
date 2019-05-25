#ifndef CALIBRATIONXML_H
#define CALIBRATIONXML_H

#include <xercesc/util/XMLString.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <string>
#include <sstream>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include "FWCore/Concurrency/interface/Xerces.h"

class CalibrationXML {
public:
  typedef XERCES_CPP_NAMESPACE::DOMDocument DOMDocument;
  typedef XERCES_CPP_NAMESPACE::DOMElement DOMElement;
  typedef XERCES_CPP_NAMESPACE::DOMNode DOMNode;
  typedef XERCES_CPP_NAMESPACE::HandlerBase HandlerBase;
  typedef XERCES_CPP_NAMESPACE::XercesDOMParser XercesDOMParser;
  typedef XERCES_CPP_NAMESPACE::XMLPlatformUtils XMLPlatformUtils;
  typedef XERCES_CPP_NAMESPACE::XMLString XMLString;

  CalibrationXML();
  ~CalibrationXML();

  /**
	* Open an XML file
	*/
  void openFile(const std::string &xmlFileName);

  /**
	* Save DOM to file
	*/
  void saveFile(const std::string &xmlFileName);

  void closeFile() {
    delete errHandler;
    delete parser;
    cms::concurrency::xercesTerminate();
    errHandler = nullptr;
    parser = nullptr;
  }
  /**
	* Return the root DOM Element of the opened XML calibration file
	*/
  DOMElement *calibrationDOM() { return m_calibrationDOM; }

  //Static function to make everything easier, less transcode and type conversion
  /**
	* Helper static function to write an attribute in a DOM Element
	*/
  template <class T>
  static void writeAttribute(DOMElement *dom, const std::string &name, const T &value) {
    std::ostringstream buffer;
    buffer << value;
    XMLCh *nameStr = XMLString::transcode(name.c_str());
    XMLCh *valueStr = XMLString::transcode(buffer.str().c_str());
    dom->setAttribute(nameStr, valueStr);
    XMLString::release(&nameStr);
    XMLString::release(&valueStr);
  }

  /**
        * Helper static function to read an attribute in a DOM Element
        */
  template <class T>
  static T readAttribute(DOMElement *dom, const std::string &name) {
    XMLCh *nameStr = XMLString::transcode(name.c_str());
    char *valueStr = XMLString::transcode(dom->getAttribute(nameStr));
    std::istringstream buffer(valueStr);
    T value;
    buffer >> value;
    XMLString::release(&nameStr);
    XMLString::release(&valueStr);
    return value;
  }

  /**
	* Helper static function to add a child in a DOM Element with indentation
	*/
  static DOMElement *addChild(DOMNode *dom, const std::string &name);

private:
  std::string m_xmlFileName;
  DOMElement *m_calibrationDOM;
  DOMDocument *doc;
  HandlerBase *errHandler;
  XercesDOMParser *parser;
};
#endif
