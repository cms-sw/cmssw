#ifndef x_SaxToDom_h
#define x_SaxToDom_h

#include <xercesc/util/XercesDefs.hpp>
#include "xercesc/sax2/DefaultHandler.hpp"
#include "DetectorDescription/Core/interface/adjgraph.h"
#include "DetectorDescription/Core/interface/graphwalker.h"

#include "DetectorDescription/RegressionTest/src/TinyDom.h"

#include <string>
#include <map>
#include <vector>

class SaxToDom : public XERCES_CPP_NAMESPACE::DefaultHandler
{
public:
  typedef XERCES_CPP_NAMESPACE::Attributes Attributes;
  typedef XERCES_CPP_NAMESPACE::SAXParseException SAXParseException;
  SaxToDom();
  ~SaxToDom();
  void startElement(const XMLCh* const uri, const XMLCh* const localname, const XMLCh* const qname, const Attributes& attrs);
  //void startElement(const XMLCh* const name, AttributeList& attributes);
  void endElement(const XMLCh* const uri, 
                            const XMLCh* const name, 
			       const XMLCh* const qname);
  const TinyDom & dom() const;

  // errors
  void error(const SAXParseException& e);
  
private:
  std::vector<NodeName> parent_;
  TinyDom dom_; 
};

#endif
