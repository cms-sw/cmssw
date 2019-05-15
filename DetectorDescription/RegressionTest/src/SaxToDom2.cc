#include "DetectorDescription/RegressionTest/src/SaxToDom2.h"
#include "DetectorDescription/RegressionTest/src/TagName.h"
#include <xercesc/util/XMLString.hpp>

#include <iostream>
#include <map>
#include <string>

using namespace std;

XERCES_CPP_NAMESPACE_USE

SaxToDom2::SaxToDom2() {
  AttList2 al;
  al[TagName("name")] = TagName("myTinyDomTest");
  Node2 nm(TagName("TinyDom2"), al);
  parent_.emplace_back(nm);
}

SaxToDom2::~SaxToDom2() {}

const TinyDom2& SaxToDom2::dom() const { return dom_; }

void SaxToDom2::startElement(const XMLCh* const uri,
                             const XMLCh* const name,
                             const XMLCh* const qname,
                             const Attributes& atts) {
  char* strx = XMLString::transcode(name);  // element-name
  AttList2 al;

  for (unsigned int i = 0; i < atts.getLength(); ++i) {
    char* aname = XMLString::transcode(atts.getLocalName(i));
    char* value = XMLString::transcode(atts.getValue(i));
    al[TagName(aname)] = TagName(value);
    XMLString::release(&aname);
    XMLString::release(&value);
  }

  // add the new element to the dom-tree
  Node2 nm(TagName(strx), al);
  Node2 par = parent_.back();
  dom_.addEdge(par, nm, AnotherDummy2());

  parent_.emplace_back(nm);
  XMLString::release(&strx);
}

void SaxToDom2::endElement(const XMLCh* const uri, const XMLCh* const name, const XMLCh* const qname) {
  parent_.pop_back();
}

// error handling
void SaxToDom2::error(const SAXParseException& e) {
  char* id = XMLString::transcode(e.getSystemId());
  char* message = XMLString::transcode(e.getMessage());
  cerr << "\nError at file " << id << ", line " << e.getLineNumber() << ", char " << e.getColumnNumber()
       << "\n  Message: " << message << endl;
  XMLString::release(&id);
  XMLString::release(&message);
}
