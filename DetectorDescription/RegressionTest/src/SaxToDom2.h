#ifndef x_SaxToDom2_h
#define x_SaxToDom2_h

#include <xercesc/util/XercesDefs.hpp>
#include "xercesc/sax2/DefaultHandler.hpp"
#include "DetectorDescription/Core/interface/adjgraph.h"
#include "DetectorDescription/Core/interface/graphwalker.h"

#include "DetectorDescription/RegressionTest/src/TinyDom2.h"

#include <string>
#include <map>
#include <vector>

class AttributeList;

class SaxToDom2 : public XERCES_CPP_NAMESPACE::DefaultHandler
{

public:
  typedef XERCES_CPP_NAMESPACE::Attributes Attributes;
  typedef XERCES_CPP_NAMESPACE::SAXParseException SAXParseException;
  SaxToDom2();
  ~SaxToDom2();
  void startElement(const XMLCh* const uri, const XMLCh* const localname, const XMLCh* const qname, const Attributes& attrs);
  //void startElement(const XMLCh* const name, AttributeList& attributes);
  void endElement(const XMLCh* const uri, 
                            const XMLCh* const name, 
			       const XMLCh* const qname);
  const TinyDom2 & dom() const;

  // errors
  void error(const SAXParseException& e);
  
private:
  std::vector<Node2> parent_;
  TinyDom2 dom_; 
};

#endif
