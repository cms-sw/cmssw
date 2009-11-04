/***************************************************************************
                          DDLSAX2FileHandler.cc  -  description
                             -------------------
    begin                : Tue Oct 23 2001
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
#include "DetectorDescription/Parser/interface/DDLSAX2FileHandler.h"
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "StrX.h"
#include "DDLElementRegistry.h"
#include "DDXMLElement.h"

// DDCore dependencies
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Base/interface/DDException.h"
#include "DetectorDescription/Core/interface/DDConstant.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"

// Xerces C++ dependencies
#include <xercesc/sax/SAXParseException.hpp>
#include <xercesc/sax/SAXException.hpp>

#include <iostream>

// ---------------------------------------------------------------------------
//  DDLSAX2Handler: Constructors and Destructor
// ---------------------------------------------------------------------------
DDLSAX2FileHandler::DDLSAX2FileHandler()
{
  createDDConstants();
  namesMap_.push_back("*** root ***");
  names_.push_back(namesMap_.size() - 1);
}

DDLSAX2FileHandler::~DDLSAX2FileHandler()
{ }

// ---------------------------------------------------------------------------
//  DDLSAX2Handler: Implementation of the SAX DocumentHandler interface
// ---------------------------------------------------------------------------
void DDLSAX2FileHandler::startElement(const XMLCh* const uri
				      , const XMLCh* const localname
				      , const XMLCh* const qname
				      , const Attributes& attrs)
{

  DCOUT_V('P', "DDLSAX2FileHandler::startElement started");
  
  std::string myElementName(StrX(qname).localForm());
  size_t i = 0;
  for ( ; i < namesMap_.size(); ++i) {
    if ( myElementName == namesMap_.at(i) ) {
      names_.push_back(i);
      break;
    }
  }
  if (i >= namesMap_.size()) {
    namesMap_.push_back(myElementName);
    names_.push_back(namesMap_.size() - 1);
  }

  ++elementTypeCounter_[myElementName];
  DDXMLElement* myElement = DDLElementRegistry::getElement(myElementName);

  unsigned int numAtts = attrs.getLength();
  std::vector<std::string> attrNames, attrValues;

  for (unsigned int i = 0; i < numAtts; ++i)
    {
      attrNames.push_back(std::string(StrX(attrs.getLocalName(i)).localForm()));
      attrValues.push_back(std::string(StrX(attrs.getValue(i)).localForm()));
    }

  DDLParser* beingParsed = DDLParser::instance();
  std::string nmspace = getnmspace(extractFileName( beingParsed->getCurrFileName()));
  myElement->loadAttributes(myElementName, attrNames, attrValues, nmspace);
  //  initialize text
  myElement->loadText(std::string()); 
  DCOUT_V('P', "DDLSAX2FileHandler::startElement completed");
}

void DDLSAX2FileHandler::endElement(const XMLCh* const uri
				    , const XMLCh* const localname
				    , const XMLCh* const qname)
{
  std::string myElementName = namesMap_.at(names_.at(names_.size() -1));

  DCOUT_V('P', "DDLSAX2FileHandler::endElement started");
  DCOUT_V('P', "    " + myElementName);

  DDXMLElement* myElement = DDLElementRegistry::getElement(myElementName);
  DDLParser* beingParsed = DDLParser::instance();
  std::string nmspace = getnmspace(extractFileName( beingParsed->getCurrFileName()));
  // The need for processElement to have the nmspace so that it can 
  // do the necessary gymnastics made things more complicated in the
  // effort to allow fully user-controlled namespaces.  So the "magic"
  // trick of setting nmspace to "!" is used :(... I don't like this magic trick
  // -- Michael Case 2008-11-06
  if ( userNS_ ) {
    nmspace = "!";
  }
  DDCurrentNamespace::ns() = nmspace;
  myElement->processElement(myElementName, nmspace);
  DCOUT_V('P', "DDLSAX2FileHandler::endElement completed");
  names_.pop_back();
}

void DDLSAX2FileHandler::characters(  const   XMLCh* const    chars
				    , const unsigned int    length)
{
  DCOUT_V('P', "DDLSAX2FileHandler::characters started");

  DDXMLElement* myElement = NULL;
  myElement = DDLElementRegistry::getElement(namesMap_.at(names_.at(names_.size() -1)));

  std::string inString = "";
  for (unsigned int i = 0; i < length; ++i)
    {
      char s = chars[i];
      inString = inString + s;
    }
  if (myElement->gotText())
    myElement->appendText(inString);
  else
    myElement->loadText(inString);

  DCOUT_V('P', "DDLSAX2FileHandler::characters completed"); 
}

void DDLSAX2FileHandler::comment( const   XMLCh* const    chars
				  , const unsigned int    length)
{
  // ignore, discard, overkill since base class also has this...
}

std::string DDLSAX2FileHandler::extractFileName(std::string fullname)
{
  std::string ret = "";
  size_t bit = fullname.rfind('/');
  if ( bit < fullname.size() - 2 ) {
    ret=fullname.substr(bit+1);
  }
  return ret;
}

void DDLSAX2FileHandler::dumpElementTypeCounter()
{
    //for (std::map<std::string, int>::const_iterator it = elementTypeCounter_.begin();
    //   it != elementTypeCounter_.end(); ++it)
    // There used to be cout printout here. No longer.
}

void DDLSAX2FileHandler::createDDConstants() const
{
  DDConstant::createConstantsFromEvaluator();
}

const std::string& DDLSAX2FileHandler::parent() const
{
  if (names_.size() > 1)
    {
      return namesMap_.at(names_.size() - 2);
    }
  return namesMap_[0];
}

const std::string& DDLSAX2FileHandler::self() const
{
  if (names_.size() > 2)
    return namesMap_.at(names_.size() - 1);
  return namesMap_[0];
}
