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
#include <xercesc/sax2/Attributes.hpp>
#include <xercesc/sax/SAXParseException.hpp>
#include <xercesc/sax/SAXException.hpp>

// Seal
#include "SealUtil/TimingReport.h"

#include <iostream>
#include <vector>
#include <string>

// ---------------------------------------------------------------------------
//  DDLSAX2Handler: Constructors and Destructor
// ---------------------------------------------------------------------------
DDLSAX2FileHandler::DDLSAX2FileHandler() : t_()
{
  createDDConstants();
  std::string* sp = new std::string("*** root ***");
  namesMap_[*sp] = sp;
  names_.push_back(sp);
  
}

DDLSAX2FileHandler::~DDLSAX2FileHandler()
{
  t_.dump();
}

// ---------------------------------------------------------------------------
//  DDLSAX2Handler: Implementation of the SAX DocumentHandler interface
// ---------------------------------------------------------------------------
void DDLSAX2FileHandler::startElement(const XMLCh* const uri
				      , const XMLCh* const localname
				      , const XMLCh* const qname
				      , const Attributes& attrs)
{
  // static seal::SealTimer tdds2fhse("DDLSAX2FileHandler::startElement(..)");
  t_.item(std::string("DDLSAX2FileHandler::startElement(...)")).accumulate();
  t_.item(std::string("DDLSAX2FileHandler::startElement(...)")).chrono().start();
  DCOUT_V('P', "DDLSAX2FileHandler::startElement started");
  
  std::map<std::string, std::string*>::const_iterator namePtr = namesMap_.find(std::string(StrX(qname).localForm()));
  std::string* nameInt;
  if (namePtr != namesMap_.end())
    {
      nameInt = namePtr->second;
    }
  else
    {
      std::string * sp = new std::string(StrX(qname).localForm());
      nameInt = sp;
      namesMap_[*sp] = nameInt;
    }
  names_.push_back(nameInt);
  std::string myElementName(StrX(qname).localForm());

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
  t_.item(std::string("DDLSAX2FileHandler::startElement(...)")).chrono().stop();
  DCOUT_V('P', "DDLSAX2FileHandler::startElement completed");
}

void DDLSAX2FileHandler::endElement(const XMLCh* const uri
				    , const XMLCh* const localname
				    , const XMLCh* const qname)
{
  std::string myElementName = *(names_.back());

  DCOUT_V('P', "DDLSAX2FileHandler::endElement started");
  DCOUT_V('P', "    " + myElementName);

  DDXMLElement* myElement = DDLElementRegistry::getElement(myElementName);
  DDLParser* beingParsed = DDLParser::instance();
  std::string nmspace = getnmspace(extractFileName( beingParsed->getCurrFileName()));
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
  myElement = DDLElementRegistry::getElement(*(names_.back()));

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
  for (std::map<std::string, int>::const_iterator it = elementTypeCounter_.begin();
       it != elementTypeCounter_.end(); ++it)
    std::cout << "Element: " << it->first << " (" << it->second << ")" << std::endl;
}

void DDLSAX2FileHandler::createDDConstants() const
{
  DDConstant::createConstantsFromEvaluator();
}

std::string& DDLSAX2FileHandler::parent() 
{
  if (names_.size() > 1)
    {
      return *(names_[names_.size() - 2]);
    }
  return *(names_[0]);
}

std::string& DDLSAX2FileHandler::self()
{
  if (names_.size() > 2)
    return *(names_[names_.size() - 1]);
  return *(names_[0]);
}
