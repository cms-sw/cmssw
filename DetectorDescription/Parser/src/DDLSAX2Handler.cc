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

// ---------------------------------------------------------------------------
//  Includes
// ---------------------------------------------------------------------------
#include "DetectorDescription/Parser/interface/DDLSAX2Handler.h"
#include "DetectorDescription/Parser/interface/StrX.h"
#include "DetectorDescription/Base/interface/DDException.h"

// Xerces C++ dependencies
#include <xercesc/sax2/Attributes.hpp>
#include <xercesc/sax/SAXParseException.hpp>
#include <xercesc/sax/SAXException.hpp>

#include <iostream>
#include <vector>
#include <string>

// ---------------------------------------------------------------------------
//  DDLSAX2Handler: Constructors and Destructor
// ---------------------------------------------------------------------------
DDLSAX2Handler::DDLSAX2Handler() :

  attrCount_(0)
  , characterCount_(0)
  , elementCount_(0)
  , spaceCount_(0)
  , sawErrors_(false)
{
  //  std::cout << " DDLSAX2Handler constructed " << std::endl;
}

DDLSAX2Handler::~DDLSAX2Handler()
{
  //  std::cout << " DDLSAX2Handler destructed " << std::endl;
}

// ---------------------------------------------------------------------------
//  DDLSAX2Handler: Implementation of the SAX DocumentHandler interface
// ---------------------------------------------------------------------------

void DDLSAX2Handler::startElement(const XMLCh* const uri
				  , const XMLCh* const localname
				  , const XMLCh* const qname
				  , const Attributes& attrs)
{
  ++elementCount_;
  attrCount_ += attrs.getLength();
}

void DDLSAX2Handler::endElement(const XMLCh* const uri
				    , const XMLCh* const localname
				    , const XMLCh* const qname)
{
  // do nothing
}


void DDLSAX2Handler::characters(  const   XMLCh* const    chars
				  , const unsigned int    length)
{
  characterCount_ += length;
}

void DDLSAX2Handler::comment( const XMLCh *const chars, const unsigned int length )
{
  // do nothing default..
}

void DDLSAX2Handler::ignorableWhitespace( const   XMLCh* const chars
					  , const unsigned int length)
{
  spaceCount_ += length;
}

void DDLSAX2Handler::resetDocument()
{
  attrCount_ = 0;
  characterCount_ = 0;
  elementCount_ = 0;
  spaceCount_ = 0;
}

void DDLSAX2Handler::dumpStats(const std::string& fname)
{

  std::cout << "DDLSAX2Handler::dumpStats, file: " 
       << fname << " ("
       << getElementCount() << " elems, "
       << getAttrCount() << " attrs, "
       << getSpaceCount() << " spaces, "
       << getCharacterCount() << " chars)" << std::endl;

}

// ---------------------------------------------------------------------------
//  DDLSAX2Handler: Overrides of the SAX ErrorHandler interface
// ---------------------------------------------------------------------------
void DDLSAX2Handler::error(const SAXParseException& e)
{
  sawErrors_ = true;
  std::cout << "\nError at file " << StrX(e.getSystemId())
       << ", line " << e.getLineNumber()
       << ", char " << e.getColumnNumber()
       << "\n  Message: " << StrX(e.getMessage()) << std::endl;
}

void DDLSAX2Handler::fatalError(const SAXParseException& e)
{
  sawErrors_ = true;
  std::cout << "\nFatal Error at file " << StrX(e.getSystemId())
       << ", line " << e.getLineNumber()
       << ", char " << e.getColumnNumber()
       << "\n  Message: " 
    << StrX(e.getMessage()) << std::endl;
//   std::cout << "Continue (1=yes,0=no)? ";
//   int i;
//   cin >> i;
//  if (!i)
    throw DDException(std::string("Unrecoverable Error: ") + std::string(StrX(e.getMessage()).localForm()));     
}

void DDLSAX2Handler::warning(const SAXParseException& e)
{
  std::cout << "\nWarning at file " << StrX(e.getSystemId())
       << ", line " << e.getLineNumber()
       << ", char " << e.getColumnNumber()
       << "\n  Message: " << StrX(e.getMessage()) << std::endl;
}

std::string DDLSAX2Handler::getnmspace(const std::string& fname)
{
  size_t j = 0;
  std::string ret="";
  while (j < fname.size() && fname[j] != '.')
    ++j;
  if (j < fname.size() && fname[j] == '.')
    ret = fname.substr(0, j);
  return ret;
}
