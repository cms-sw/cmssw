#ifndef _MISCALIB_READER_FROM_XML_DOM_UTILS_H
#define _MISCALIB_READER_FROM_XML_DOM_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/dom/DOMNode.hpp>

/** a collection of some XML reading utilities */
class MiscalibReaderFromXMLDomUtils
{
public:
  inline static std::string toString(const XMLCh *str);

  inline static int getIntAttribute(XERCES_CPP_NAMESPACE::DOMNamedNodeMap *attributes, std::string attr_name, bool &well_formed_string);
  
  inline static double getFloatAttribute(XERCES_CPP_NAMESPACE::DOMNamedNodeMap *attributes, std::string attr_name, bool &well_formed_string);

  inline static std::string getStringAttribute(XERCES_CPP_NAMESPACE::DOMNamedNodeMap *attributes, std::string attr_name);

  inline static bool hasAttribute(XERCES_CPP_NAMESPACE::DOMNamedNodeMap *attributes, std::string attr_name);
  
};

#include <CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLDomUtils.icc>

#endif
