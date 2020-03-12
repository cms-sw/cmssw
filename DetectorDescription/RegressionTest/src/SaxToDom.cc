#include "DetectorDescription/RegressionTest/src/SaxToDom.h"

#include <iostream>
#include <map>
#include <string>
#include <xercesc/util/XMLString.hpp>

using namespace std;

XERCES_CPP_NAMESPACE_USE

SaxToDom::SaxToDom() { parent_.emplace_back(NodeName("TinyDom")); }

SaxToDom::~SaxToDom() {}

const TinyDom& SaxToDom::dom() const { return dom_; }

void SaxToDom::startElement(const XMLCh* const uri,
                            const XMLCh* const name,
                            const XMLCh* const qname,
                            const Attributes& atts) {
  char* strx = XMLString::transcode(name);  // element-name
  NodeName nm(strx);                        // as a temp.string

  AttList al;  // map of attributes -> values
  for (unsigned int i = 0; i < atts.getLength(); ++i) {
    char* aname = XMLString::transcode(atts.getLocalName(i));
    char* value = XMLString::transcode(atts.getValue(i));
    // fill the tiny-dom-attribute-list (i.e. the map)
    al[NodeName(aname)] = NodeName(value);

    XMLString::release(&aname);
    XMLString::release(&value);
  }
  // add the new element to the dom-tree
  dom_.addEdge(parent_.back(), nm, al);

  // set the parent_ to the actual node
  parent_.emplace_back(nm);
  XMLString::release(&strx);
}

void SaxToDom::endElement(const XMLCh* const uri, const XMLCh* const name, const XMLCh* const qname) {
  parent_.pop_back();
}

// error handling
void SaxToDom::error(const SAXParseException& e) {
  char* id = XMLString::transcode(e.getSystemId());
  char* message = XMLString::transcode(e.getMessage());
  cerr << "\nError at file " << id << ", line " << e.getLineNumber() << ", char " << e.getColumnNumber()
       << "\n  Message: " << message << endl;
  XMLString::release(&id);
  XMLString::release(&message);
}
