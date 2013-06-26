// $Id: XHTMLMaker.cc,v 1.12 2011/07/07 09:22:45 mommsen Exp $
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
using namespace stor;

namespace
{
  // String to XMLCh: Note that the Xerces-C++ documentation says you
  // have to call XMLString::release(p) for every pointer p returned
  // from XMLString::transcode().
  inline XMLCh* xs_( const std::string& str )
  {
    // std::string::data() is not required to return a null-terminated
    // byte array; c_str() is required to do so.
    return xercesc::XMLString::transcode(str.c_str());
  }
  
  inline XMLCh* xs_(const char* str)
  {
    return xercesc::XMLString::transcode(str);
  }
}


XHTMLMaker::XHTMLMaker()
{
  XMLCh* xls = xs_("ls");
  DOMImplementation* imp =
    DOMImplementationRegistry::getDOMImplementation(xls);

  XMLCh* xhtml_s =xs_("html") ;
  XMLCh* p_id = xs_( "-//W3C//DTD XHTML 1.0 Strict//EN" );
  XMLCh* s_id = xs_( "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd" );

  typ_ = imp->createDocumentType( xhtml_s, p_id, s_id );

  XMLCh* ns_uri = xs_( "http://www.w3.org/1999/xhtml" );

  doc_ = imp->createDocument( ns_uri, xhtml_s, typ_ );


  if( !doc_ )
    {
      std::cerr << "Cannot create document" << std::endl;
      return;
    }

  XMLCh* encoding = xs_("utf-8");
  doc_->setEncoding(encoding);
  //doc_->setStandalone( true );
  XMLCh* version = xs_("1.0");
  doc_->setVersion(version);

  pageStarted_ = false;

  writer_ =
    ( (DOMImplementationLS*)imp )->createDOMWriter();

  XMLString::release(&xls);
  XMLString::release(&xhtml_s);
  XMLString::release(&p_id);
  XMLString::release(&s_id);
  XMLString::release(&ns_uri);
  XMLString::release(&encoding);
  XMLString::release(&version);

  if( !writer_ )
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
  delete doc_;
  delete writer_;
  delete typ_;
}

///////////////////////////////////////
//// Initialize page, return body: ////
///////////////////////////////////////

XHTMLMaker::Node* XHTMLMaker::start( const std::string& title )
{

  if( pageStarted_ )
    {
      std::cerr << "Page already started" << std::endl;
      return (Node*)0;
    }

  pageStarted_ = true;

  // Root element:
  Node* el_xhtml = doc_->getDocumentElement();

  // Head:
  XMLCh* h = xs_("head");
  head_ = doc_->createElement(h);
  XMLString::release(&h);
  el_xhtml->appendChild( head_ );


  // Title:
  XMLCh* xtitle = xs_("title");
  Node* el_title = doc_->createElement(xtitle);
  XMLString::release(&xtitle);
  head_->appendChild( el_title );
  xtitle = xs_(title);
  DOMText* txt_title = doc_->createTextNode(xtitle);
  XMLString::release(&xtitle);
  el_title->appendChild( txt_title );

  // Body:
  XMLCh* xbody = xs_("body");
  Node* el_body = doc_->createElement(xbody);
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
  XMLCh* xname = xs_(name);
  Node* el = doc_->createElement(xname);
  XMLString::release(&xname);
  parent->appendChild( el );

  for( AttrMap::const_iterator i = attrs.begin(); i != attrs.end();
	 ++i )
    {
      XMLCh* xfirst = xs_(i->first);
      XMLCh* xsecond = xs_(i->second);
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
  XMLCh* xdata = xs_(data);
  DOMText* txt = doc_->createTextNode(xdata);
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

/////////////////////////////////////
//// Add a unsigned long as hex: ////
/////////////////////////////////////
void XHTMLMaker::addHex( Node* parent, const unsigned long& value )
{
    ostringstream tmpString;
    tmpString << "0x" << std::hex << value;
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

//////////////////////////////
//// Add a boolean value: ////
//////////////////////////////
void XHTMLMaker::addBool( Node* parent, const bool& value )
{
    addText( parent, value ? "True" : "False" );
}

/////////////////////////////////
//// Set DOMWriter features: ////
/////////////////////////////////

void XHTMLMaker::setWriterFeatures_()
{

  //writer_->setNewLine( (const XMLCh*)( L"\n" ) );

  if( writer_->canSetFeature( XMLUni::fgDOMWRTSplitCdataSections, true ) )
    {
      writer_->setFeature( XMLUni::fgDOMWRTSplitCdataSections, true );
    }

  if( writer_->canSetFeature( XMLUni::fgDOMWRTDiscardDefaultContent, true ) )
    {
      writer_->setFeature( XMLUni::fgDOMWRTDiscardDefaultContent, true );
    }

  if( writer_->canSetFeature( XMLUni::fgDOMWRTFormatPrettyPrint, true ) )
    {
      writer_->setFeature( XMLUni::fgDOMWRTFormatPrettyPrint, true );
    }

  if( writer_->canSetFeature( XMLUni::fgDOMWRTBOM, true ) )
    {
      writer_->setFeature( XMLUni::fgDOMWRTBOM, true );
    }

}

//////////////////////////////
//// Dump page to stdout: ////
//////////////////////////////
void XHTMLMaker::out()
{
  setWriterFeatures_();
  XMLFormatTarget* ftar = new StdOutFormatTarget();
  fflush( stdout );
  writer_->writeNode( ftar, *doc_ );
  delete ftar;
}

////////////////////////////////////
//// Dump page to a local file: ////
////////////////////////////////////
void XHTMLMaker::out( const std::string& filename )
{
  setWriterFeatures_();
  XMLCh* xfilename = xs_(filename);
  XMLFormatTarget* ftar = new LocalFileFormatTarget(xfilename);
  writer_->writeNode( ftar, *doc_ );
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
  setWriterFeatures_();
  MemBufFormatTarget* ftar = new MemBufFormatTarget();
  writer_->writeNode( ftar, *doc_ );
  dest << ftar->getRawBuffer();
  delete ftar;
}

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
