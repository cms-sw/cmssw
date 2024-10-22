#ifndef DETECTOR_DESCRIPTION_REGRESSION_TEST_SAXTODOM2_H
#define DETECTOR_DESCRIPTION_REGRESSION_TEST_SAXTODOM2_H

#include <xercesc/util/XercesDefs.hpp>
#include <map>
#include <string>
#include <vector>

#include "DataFormats/Math/interface/Graph.h"
#include "DataFormats/Math/interface/GraphWalker.h"
#include "DetectorDescription/RegressionTest/src/TinyDom2.h"
#include "xercesc/sax/SAXParseException.hpp"
#include "xercesc/sax2/Attributes.hpp"
#include "xercesc/sax2/DefaultHandler.hpp"
#include "xercesc/util/XercesVersion.hpp"

class AttributeList;

class SaxToDom2 : public XERCES_CPP_NAMESPACE::DefaultHandler {
public:
  using Attributes = XERCES_CPP_NAMESPACE::Attributes;
  using SAXParseException = XERCES_CPP_NAMESPACE::SAXParseException;
  SaxToDom2();
  ~SaxToDom2() override;
  void startElement(const XMLCh* uri, const XMLCh* localname, const XMLCh* qname, const Attributes& attrs) override;
  void endElement(const XMLCh* uri, const XMLCh* name, const XMLCh* qname) override;
  const TinyDom2& dom() const;

  // errors
  void error(const SAXParseException& e) override;

private:
  std::vector<Node2> parent_;
  TinyDom2 dom_;
};

#endif
