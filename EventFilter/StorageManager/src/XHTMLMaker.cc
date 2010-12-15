// $Id: XHTMLMaker.cc,v 1.9 2010/12/15 10:09:14 mommsen Exp $
/// @file: XHTMLMaker.cc

#include "EventFilter/StorageManager/interface/XHTMLMaker.h"

#include <sstream>
#include <iomanip>
#include <iostream>
#include <cstdio>
#include <xercesc/framework/StdOutFormatTarget.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/framework/MemBufFormatTarget.hpp>

using namespace std;
using namespace xercesc;

namespace
{
  // String to XMLCh: Note that the Xerces-C++ documentation says you
  // have to call XMLString::release(p) for every pointer p returned
  // from XMLString::transcode().
  inline XMLCh* _xs( const std::string& str )
  {
    // std::string::data() is not required to return a null-terminated
    // byte array; c_str() is required to do so.
    return xercesc::XMLString::transcode(str.c_str());
  }
  
  inline XMLCh* _xs(const char* str)
  {
    return xercesc::XMLString::transcode(str);
  }
}


XHTMLMaker::XHTMLMaker()
{
  XMLCh* xls = _xs("ls");
  DOMImplementation* imp =
    DOMImplementationRegistry::getDOMImplementation(xls);

  XMLCh* xhtml_s =_xs("html") ;
  XMLCh* p_id = _xs( "-//W3C//DTD XHTML 1.0 Strict//EN" );
  XMLCh* s_id = _xs( "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd" );

  _typ = imp->createDocumentType( xhtml_s, p_id, s_id );

  XMLCh* ns_uri = _xs( "http://www.w3.org/1999/xhtml" );

  _doc = imp->createDocument( ns_uri, xhtml_s, _typ );


  if( !_doc )
    {
      std::cerr << "Cannot create document" << std::endl;
      return;
    }

  XMLCh* encoding = _xs("utf-8");
  _doc->setEncoding(encoding);
  //_doc->setStandalone( true );
  XMLCh* version = _xs("1.0");
  _doc->setVersion(version);

  _page_started = false;

  _writer =
    ( (DOMImplementationLS*)imp )->createDOMWriter();

  XMLString::release(&xls);
  XMLString::release(&xhtml_s);
  XMLString::release(&p_id);
  XMLString::release(&s_id);
  XMLString::release(&ns_uri);
  XMLString::release(&encoding);
  XMLString::release(&version);

  if( !_writer )
    {
      std::cerr << "Cannot create DOM writer" << std::endl;
      return;
    }
}

/////////////////////
//// Destructor: ////
/////////////////////

XHTMLMaker::~XHTMLMaker()
{
  delete _doc;
  delete _writer;
  delete _typ;
}

///////////////////////////////////////
//// Initialize page, return body: ////
///////////////////////////////////////

XHTMLMaker::Node* XHTMLMaker::start( const std::string& title )
{

  if( _page_started )
    {
      std::cerr << "Page already started" << std::endl;
      return (Node*)0;
    }

  _page_started = true;

  // Root element:
  Node* el_xhtml = _doc->getDocumentElement();

  // Head:
  XMLCh* h = _xs("head");
  _head = _doc->createElement(h);
  XMLString::release(&h);
  el_xhtml->appendChild( _head );


  // Title:
  XMLCh* xtitle = _xs("title");
  Node* el_title = _doc->createElement(xtitle);
  XMLString::release(&xtitle);
  _head->appendChild( el_title );
  xtitle = _xs(title);
  DOMText* txt_title = _doc->createTextNode(xtitle);
  XMLString::release(&xtitle);
  el_title->appendChild( txt_title );

  // Body:
  XMLCh* xbody = _xs("body");
  Node* el_body = _doc->createElement(xbody);
  XMLString::release(&xbody);
  el_xhtml->appendChild( el_body );

  return el_body;

}

////////////////////////////
//// Add child element: ////
////////////////////////////

XHTMLMaker::Node* XHTMLMaker::addNode( const std::string& name,
				       XHTMLMaker::Node* parent,
				       const AttrMap& attrs )
{
  XMLCh* xname = _xs(name);
  Node* el = _doc->createElement(xname);
  XMLString::release(&xname);
  parent->appendChild( el );

  for( AttrMap::const_iterator i = attrs.begin(); i != attrs.end();
	 ++i )
    {
      XMLCh* xfirst = _xs(i->first);
      XMLCh* xsecond = _xs(i->second);
      el->setAttribute(xfirst, xsecond);
      XMLString::release(&xfirst);
      XMLString::release(&xsecond);
    }

  return el;

}

///////////////////
//// Add text: ////
///////////////////

void XHTMLMaker::addText( Node* parent, const std::string& data )
{
  XMLCh* xdata = _xs(data);
  DOMText* txt = _doc->createTextNode(xdata);
  XMLString::release(&xdata);
  parent->appendChild( txt );
}

/////////////////////
//// Add an int: ////
/////////////////////
void XHTMLMaker::addInt( Node* parent, const int& value )
{
    ostringstream tmpString;
    tmpString << value;
    addText( parent, tmpString.str() );
}

//////////////////////////////
//// Add an unsigned int: ////
//////////////////////////////
void XHTMLMaker::addInt( Node* parent, const unsigned int& value )
{
    ostringstream tmpString;
    tmpString << value;
    addText( parent, tmpString.str() );
}

/////////////////////
//// Add a long: ////
/////////////////////
void XHTMLMaker::addInt( Node* parent, const long& value )
{
    ostringstream tmpString;
    tmpString << value;
    addText( parent, tmpString.str() );
}

///////////////////////////////
//// Add an unsigned long: ////
///////////////////////////////
void XHTMLMaker::addInt( Node* parent, const unsigned long& value )
{
    ostringstream tmpString;
    tmpString << value;
    addText( parent, tmpString.str() );
}

//////////////////////////
//// Add a long long: ////
//////////////////////////
void XHTMLMaker::addInt( Node* parent, const long long& value )
{
    ostringstream tmpString;
    tmpString << value;
    addText( parent, tmpString.str() );
}

////////////////////////////////////
//// Add an unsigned long long: ////
////////////////////////////////////
void XHTMLMaker::addInt( Node* parent, const unsigned long long& value )
{
    ostringstream tmpString;
    tmpString << value;
    addText( parent, tmpString.str() );
}

/////////////////////////////
//// Add a double value: ////
/////////////////////////////
void XHTMLMaker::addDouble( Node* parent, const double& value, const unsigned int& precision )
{
    ostringstream tmpString;
    tmpString << fixed << std::setprecision( precision ) << value;
    addText( parent, tmpString.str() );
}

/////////////////////////////////
//// Set DOMWriter features: ////
/////////////////////////////////

void XHTMLMaker::_setWriterFeatures()
{

  //_writer->setNewLine( (const XMLCh*)( L"\n" ) );

  if( _writer->canSetFeature( XMLUni::fgDOMWRTSplitCdataSections, true ) )
    {
      _writer->setFeature( XMLUni::fgDOMWRTSplitCdataSections, true );
    }

  if( _writer->canSetFeature( XMLUni::fgDOMWRTDiscardDefaultContent, true ) )
    {
      _writer->setFeature( XMLUni::fgDOMWRTDiscardDefaultContent, true );
    }

  if( _writer->canSetFeature( XMLUni::fgDOMWRTFormatPrettyPrint, true ) )
    {
      _writer->setFeature( XMLUni::fgDOMWRTFormatPrettyPrint, true );
    }

  if( _writer->canSetFeature( XMLUni::fgDOMWRTBOM, true ) )
    {
      _writer->setFeature( XMLUni::fgDOMWRTBOM, true );
    }

}

//////////////////////////////
//// Dump page to stdout: ////
//////////////////////////////
void XHTMLMaker::out()
{
  _setWriterFeatures();
  XMLFormatTarget* ftar = new StdOutFormatTarget();
  fflush( stdout );
  _writer->writeNode( ftar, *_doc );
  delete ftar;
}

////////////////////////////////////
//// Dump page to a local file: ////
////////////////////////////////////
void XHTMLMaker::out( const std::string& filename )
{
  _setWriterFeatures();
  XMLCh* xfilename = _xs(filename);
  XMLFormatTarget* ftar = new LocalFileFormatTarget(xfilename);
  _writer->writeNode( ftar, *_doc );
  XMLString::release(&xfilename);
  delete ftar;
}

////////////////////////////////////
//// Dump the page to a string: ////
////////////////////////////////////
void XHTMLMaker::out( std::string& dest )
{
   std::ostringstream stream;
   out( stream );
   dest = stream.str();
}

//////////////////////////////////////////////
//// Dump the page into an output stream: ////
//////////////////////////////////////////////
void XHTMLMaker::out( std::ostream& dest )
{
  _setWriterFeatures();
  MemBufFormatTarget* ftar = new MemBufFormatTarget();
  _writer->writeNode( ftar, *_doc );
  dest << ftar->getRawBuffer();
  delete ftar;
}

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
