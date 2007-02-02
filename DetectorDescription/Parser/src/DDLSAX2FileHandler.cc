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
#include "DetectorDescription/Parser/interface/StrX.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/interface/DDXMLElement.h"

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
#include <algorithm>
#include <iterator>
#include <stack>

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
  t_.item("DDLSAX2FileHandler::startElement(...)").accumulate();
  t_.item("DDLSAX2FileHandler::startElement(...)").chrono().start();
  DCOUT_V('P', "DDLSAX2FileHandler::startElement started");
  
  //***  char * buf = XMLString::transcode(qname);
  StrX myName(qname);
  //***  std::map<std::string, std::string*>::const_iterator namePtr = namesMap_.find(std::string(buf));
  std::map<std::string, std::string*>::const_iterator namePtr = namesMap_.find(myName.stringForm());
  std::string* nameInt;
  if (namePtr != namesMap_.end())
    {
      nameInt = namePtr->second;
    }
  else
    {
      std::string * sp = new std::string(myName.stringForm());
      nameInt = sp;
      namesMap_[*sp] = nameInt;
    }
  names_.push_back(nameInt);
  std::string myElementName = myName.stringForm();

  //  std::cout << "start: namesMap_ = " << *(names_.back()) << "  names_.back()= " << names_.back() << std::endl;

  //***  delete[] buf;

//    char * xmlc = XMLString::transcode(qname);
//    std::string myElementName = std::string(xmlc);
//    delete[] xmlc;
  //DDLSAX2Handler::startElement(uri, localname, qname, attrs);
  //  std::cout << "names_.top() = " << names_.top() << std::endl;
  // count types of elements.  Note, this is cumulative for all files handled by
  // this handler.
  ++elementTypeCounter_[myElementName];
  DDXMLElement* myElement = DDLElementRegistry::getElement(myElementName);

  unsigned int numAtts = attrs.getLength();
  std::vector<std::string> attrNames, attrValues;

  for (unsigned int i = 0; i < numAtts; ++i)
    {
      //***      char * buf = XMLString::transcode(attrs.getLocalName(i));
      //***      std::string myattname(buf); delete[] buf;
      //**** std::string myattname = StrX(attrs.getLocalName(i)).stringForm();
      //***      buf = XMLString::transcode(attrs.getValue(i));
      //***      std::string myvalue(buf); delete[] buf;
      //****      std::string myvalue = StrX(attrs.getValue(i)).stringForm();
      //***      buf = XMLString::transcode(attrs.getQName(i));
      //***      std::string myQName(buf); delete[] buf;
      //      std::string myQName = StrX(attrs.getQName(i)).stringForm();
      attrNames.push_back(StrX(attrs.getLocalName(i)).stringForm());//myattname);
      attrValues.push_back(StrX(attrs.getValue(i)).stringForm());//myvalue);
    }

  if (myElement != NULL)
    {
      DDLParser* beingParsed = DDLParser::instance();
      std::string nmspace = getnmspace(extractFileName( beingParsed->getCurrFileName()));
      myElement->loadAttributes(myElementName, attrNames, attrValues, nmspace);
      //  initialize text
      myElement->loadText(std::string()); 
    }

  else 
    {
      std::string s = "DDLSAX2FileHandler::startElement ERROR No pointer returned from";
      s += " DDLElementRegistry::getElement( name ).  MAJOR PROBLEM.  ";
      s += "  This should never be seen!  Element is " + myElementName;
      throw DDException(s);
    }
  t_.item("DDLSAX2FileHandler::startElement(...)").chrono().stop();
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
  if (myElement != NULL)
    {
      std::string nmspace = getnmspace(extractFileName( beingParsed->getCurrFileName()));
      DDCurrentNamespace::ns() = nmspace;
      try {
	myElement->processElement(myElementName, nmspace);
      }
      catch (DDException & e) {
        std::string s(e.what());
	s+="\nDDLSAX2FileHandler::endElement call to processElement Failed.";
	myElement->throwError(s);
      }
    }
  else 
    {
      std::string s = "DDLSAX2FileHandler::endElement ERROR No pointer returned from";
      s += " DDLElement::getElement( name ).  MAJOR PROBLEM.  ";
      s += "  This should never be seen!  Element is " + myElementName;
      throw DDException(s);
    }
  DCOUT_V('P', "DDLSAX2FileHandler::endElement completed");
  names_.pop_back();
}

void DDLSAX2FileHandler::characters(  const   XMLCh* const    chars
				    , const unsigned int    length)
{
  DCOUT_V('P', "DDLSAX2FileHandler::characters started");

  //DDLSAX2Handler::characters ( chars, length );

  DDXMLElement* myElement = NULL;
//   // how to handle first one?  DDDefinition?  maybe make it in the registry instead...
//   // for now, this will work.
//   std::string myElementName = "**defaulted**";
//   if (names_.size() == 1)
//     {
//       myElement = new DDXMLElement;
//       myElement->loadText(std::string());
//     }
//   else
//     {
      myElement = DDLElementRegistry::getElement(*(names_.back()));
      //      std::string myElementName = *(names_.back());
//     }

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
  size_t startfrom = fullname.size() - 1;
  while (startfrom > 0 && fullname[startfrom] != '/')
    --startfrom;
  if (fullname[startfrom] == '/') startfrom = startfrom + 1;
  ret = fullname.substr(startfrom, fullname.size() - startfrom);
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
  try {
    DDConstant::createConstantsFromEvaluator();
  }
  catch(DDException & e) {
    std::string msg ("caught in DDLSAX2FileHandler::createDDConstants():\n");
    msg = msg + e.what();
    throw DDException(msg);
  }
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
