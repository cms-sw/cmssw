#ifndef x_SaxToDom2_h
#define x_SaxToDom2_h

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

class SaxToDom2 : public XERCES_CPP_NAMESPACE::DefaultHandler
{

public:
  typedef XERCES_CPP_NAMESPACE::Attributes Attributes;
  typedef XERCES_CPP_NAMESPACE::SAXParseException SAXParseException;
  SaxToDom2();
  ~SaxToDom2() override;
  void startElement(const XMLCh* const uri, const XMLCh* const localname, const XMLCh* const qname, const Attributes& attrs) override;
  //void startElement(const XMLCh* const name, AttributeList& attributes);
  void endElement(const XMLCh* const uri, 
                            const XMLCh* const name, 
			       const XMLCh* const qname) override;
  const TinyDom2 & dom() const;

  // errors
  void error(const SAXParseException& e) override;
  
private:
  std::vector<Node2> parent_;
  TinyDom2 dom_; 
};

#endif
