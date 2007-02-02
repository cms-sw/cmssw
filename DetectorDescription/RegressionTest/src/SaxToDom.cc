#include "DetectorDescription/RegressionTest/src/SaxToDom.h"
#include "DetectorDescription/RegressionTest/src/StrX.h"

#include <xercesc/sax2/Attributes.hpp>
#include <xercesc/sax/SAXParseException.hpp>
#include <xercesc/sax/SAXException.hpp>

#include <iostream>
//#include <string>

using namespace std;

SaxToDom::SaxToDom() 
{ parent_.push_back(NodeName("TinyDom")); }

SaxToDom::~SaxToDom() 
{ }


const TinyDom & SaxToDom::dom() const
{
   return dom_;
}


void SaxToDom::startElement( const XMLCh* const uri, 
                            const XMLCh* const name, 
			       const XMLCh* const qname, 
			       const Attributes& atts)
{
  StrX strx(name); // element-name
  NodeName nm(string(strx.localForm())); // as a temp.string
  //parent_.push_back(nm);
  AttList al; // map of attributes -> values
  for (unsigned int i = 0; i < atts.getLength(); ++i) {
    const XMLCh* aname = atts.getLocalName(i);
    const XMLCh* value = atts.getValue(i);
    // fill the tiny-dom-attribute-list (i.e. the map)
    al[NodeName((StrX(aname).localForm()))]=NodeName(StrX(value).localForm());
    //cout << "  att=" << StrX(aname) << " val=" << StrX(value) << endl;
  }  
  // add the new element to the dom-tree
  dom_.addEdge(parent_.back(), nm , al);
  //cout << "add from=" << parent_.back().str() << " to=" << nm.str() << endl;
  // set the parent_ to the actual node
  parent_.push_back(nm);
}


void SaxToDom::endElement(const XMLCh* const uri, 
                            const XMLCh* const name, 
			       const XMLCh* const qname)
{
  parent_.pop_back();
}

// error handling
void SaxToDom::error(const SAXParseException& e)
{
    cerr << "\nError at file " << StrX(e.getSystemId())
		 << ", line " << e.getLineNumber()
		 << ", char " << e.getColumnNumber()
         << "\n  Message: " << StrX(e.getMessage()) << endl;
}


