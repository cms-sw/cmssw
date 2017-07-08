#ifndef x_SaxToDom_h
#define x_SaxToDom_h

#include <xercesc/util/XercesDefs.hpp>
#include <map>
#include <string>
#include <vector>

#include "DetectorDescription/Core/interface/adjgraph.h"
#include "DetectorDescription/Core/interface/graphwalker.h"
#include "DetectorDescription/RegressionTest/src/TinyDom.h"
#include "xercesc/sax/SAXParseException.hpp"
#include "xercesc/sax2/Attributes.hpp"
#include "xercesc/sax2/DefaultHandler.hpp"
#include "xercesc/util/XercesVersion.hpp"

class SaxToDom : public XERCES_CPP_NAMESPACE::DefaultHandler
{
public:
  typedef XERCES_CPP_NAMESPACE::Attributes Attributes;
  typedef XERCES_CPP_NAMESPACE::SAXParseException SAXParseException;
  SaxToDom();
  ~SaxToDom();
  void startElement(const XMLCh* uri, const XMLCh* localname, const XMLCh* qname, const Attributes& attrs);
  //void startElement(const XMLCh* const name, AttributeList& attributes);
  void endElement(const XMLCh* uri, 
                            const XMLCh* name, 
			       const XMLCh* qname);
  const TinyDom & dom() const;

  // errors
  void error(const SAXParseException& e);
  
private:
  std::vector<NodeName> parent_;
  TinyDom dom_; 
};

#endif
