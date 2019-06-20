#ifndef DETECTOR_DESCRIPTION_REGRESSION_TEST_SAXTODOM_H
#define DETECTOR_DESCRIPTION_REGRESSION_TEST_SAXTODOM_H

#include <xercesc/util/XercesDefs.hpp>
#include <map>
#include <string>
#include <vector>

#include "DataFormats/Math/interface/Graph.h"
#include "DataFormats/Math/interface/GraphWalker.h"
#include "DetectorDescription/RegressionTest/src/TinyDom.h"
#include "xercesc/sax/SAXParseException.hpp"
#include "xercesc/sax2/Attributes.hpp"
#include "xercesc/sax2/DefaultHandler.hpp"
#include "xercesc/util/XercesVersion.hpp"

class SaxToDom : public XERCES_CPP_NAMESPACE::DefaultHandler {
public:
  using Attributes = XERCES_CPP_NAMESPACE::Attributes;
  using SAXParseException = XERCES_CPP_NAMESPACE::SAXParseException;
  SaxToDom();
  ~SaxToDom() override;
  void startElement(const XMLCh* uri, const XMLCh* localname, const XMLCh* qname, const Attributes& attrs) override;
  void endElement(const XMLCh* uri, const XMLCh* name, const XMLCh* qname) override;
  const TinyDom& dom() const;

  // errors
  void error(const SAXParseException& e) override;

private:
  std::vector<NodeName> parent_;
  TinyDom dom_;
};

#endif
