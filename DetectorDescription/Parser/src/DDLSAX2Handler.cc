/***************************************************************************
                          DDLSAX2Handler.cc  -  description
                             -------------------
    begin                : Mon Oct 22 2001
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *           DDDParser sub-component of DDD                                *
 *                                                                         *
 ***************************************************************************/

#include "DetectorDescription/Parser/interface/DDLSAX2Handler.h"
#include "DetectorDescription/Parser/src/StrX.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

DDLSAX2Handler::DDLSAX2Handler( void )
  : attrCount_(0),
    characterCount_(0),
    elementCount_(0),
    spaceCount_(0),
    sawErrors_(false),
    userNS_(false)
{}

DDLSAX2Handler::~DDLSAX2Handler( void )
{}

// ---------------------------------------------------------------------------
//  DDLSAX2Handler: Implementation of the SAX DocumentHandler interface
// ---------------------------------------------------------------------------

void
DDLSAX2Handler::startElement( const XMLCh* const uri,
			      const XMLCh* const localname,
			      const XMLCh* const qname,
			      const Attributes& attrs )
{
  ++elementCount_;
  attrCount_ += attrs.getLength();
}

void
DDLSAX2Handler::endElement( const XMLCh* const uri,
			    const XMLCh* const localname,
			    const XMLCh* const qname )
{
  // do nothing
}

void
DDLSAX2Handler::characters( const XMLCh* const chars,
			    const unsigned int length )
{
  characterCount_ += length;
}

void
DDLSAX2Handler::comment( const XMLCh *const chars, const unsigned int length )
{
  // do nothing default..
}

void
DDLSAX2Handler::ignorableWhitespace( const XMLCh* const chars,
				     const unsigned int length )
{
  spaceCount_ += length;
}

void
DDLSAX2Handler::resetDocument( void )
{
  attrCount_ = 0;
  characterCount_ = 0;
  elementCount_ = 0;
  spaceCount_ = 0;
}

void
DDLSAX2Handler::dumpStats( const std::string& fname )
{
  std::cout << "DetectorDescription/Parser/interface/DDLSAX2Handler::dumpStats, file: " 
       << fname << " ("
       << getElementCount() << " elems, "
       << getAttrCount() << " attrs, "
       << getSpaceCount() << " spaces, "
       << getCharacterCount() << " chars)" << std::endl;
}

// ---------------------------------------------------------------------------
//  DDLSAX2Handler: Overrides of the SAX ErrorHandler interface
//  Implements ALL required by the Xerces ErrorHandler interface as of 2007-06-26.
// ---------------------------------------------------------------------------
void
DDLSAX2Handler::error( const SAXParseException& e )
{
  sawErrors_ = true;
  edm::LogError("DetectorDescription_Parser_DDLSAX2Handler") 
    << "\nError at file " << StrX(e.getSystemId())
    << ", line " << e.getLineNumber()
    << ", char " << e.getColumnNumber()
    << "\n  Message: " << StrX(e.getMessage()) << std::endl;
}

void
DDLSAX2Handler::fatalError( const SAXParseException& e )
{
  sawErrors_ = true;
  edm::LogError("DetectorDescription_Parser_DDLSAX2Handler") 
    << "\nFatal Error at file " << StrX(e.getSystemId())
    << ", line " << e.getLineNumber()
    << ", char " << e.getColumnNumber()
    << "\n  Message: " 
    << StrX(e.getMessage()) << std::endl;
  throw cms::Exception("DDException") << "DetectorDescription_Parser_Unrecoverable_Error_from_Xerces: "
    << std::string(StrX(e.getMessage()).localForm())
    << " file: " << std::string(StrX(e.getSystemId()).localForm())
    << " line: " << e.getLineNumber() << " col: " << e.getColumnNumber();
}

void
DDLSAX2Handler::warning( const SAXParseException& e )
{
  edm::LogWarning("DetectorDescription_Parser_DDLSAX2Handler") 
    << "\nWarning at file " << StrX(e.getSystemId())
    << ", line " << e.getLineNumber()
    << ", char " << e.getColumnNumber()
    << "\n  Message: " << StrX(e.getMessage()) << std::endl;
}

void
DDLSAX2Handler::setUserNS( bool userns )
{ 
  userNS_ = userns; 
}

void
DDLSAX2Handler::setNameSpace( const std::string& nms )
{
  nmspace_ = nms;
}
