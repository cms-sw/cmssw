#include "DetectorDescription/RegressionTest/src/SaxToDom2.h"
#include "DetectorDescription/RegressionTest/src/StrX.h"

#include <xercesc/sax2/Attributes.hpp>
#include <xercesc/sax/SAXParseException.hpp>
#include <xercesc/sax/SAXException.hpp>

#include <iostream>
//#include <string>

using namespace std;

SaxToDom2::SaxToDom2() 
{ 
  AttList2 al;
  al [ TagName("name") ] = TagName("myTinyDomTest");
  Node2 nm(TagName("TinyDom2"), al);
  parent_.push_back( nm );
}

SaxToDom2::~SaxToDom2() 
{ }


const TinyDom2 & SaxToDom2::dom() const
{
   return dom_;
}


void SaxToDom2::startElement( const XMLCh* const uri, 
                            const XMLCh* const name, 
			       const XMLCh* const qname, 
			       const Attributes& atts)
{
  StrX strx(name); // element-name
  AttList2 al;

  for (unsigned int i = 0; i < atts.getLength(); ++i)
    {
      const XMLCh* aname = atts.getLocalName(i);
      const XMLCh* value = atts.getValue(i);
      al[TagName((StrX(aname).localForm()))]=TagName(StrX(value).localForm());
    }

  // add the new element to the dom-tree
  Node2 nm(TagName(strx.localForm()) , al);
  Node2 par = parent_.back();
  dom_.addEdge(par, nm, AnotherDummy2());

  parent_.push_back(nm);
}


void SaxToDom2::endElement(const XMLCh* const uri, 
                            const XMLCh* const name, 
			       const XMLCh* const qname)
{
  parent_.pop_back();
}

// error handling
void SaxToDom2::error(const SAXParseException& e)
{
    cerr << "\nError at file " << StrX(e.getSystemId())
		 << ", line " << e.getLineNumber()
		 << ", char " << e.getColumnNumber()
         << "\n  Message: " << StrX(e.getMessage()) << endl;
}


