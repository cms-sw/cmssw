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
#include "DetectorDescription/DDParser/interface/DDLSAX2FileHandler.h"
#include "DetectorDescription/DDParser/interface/DDLParser.h"
#include "DetectorDescription/DDParser/interface/StrX.h"
#include "DetectorDescription/DDParser/interface/DDLElementRegistry.h"
#include "DetectorDescription/DDParser/interface/DDXMLElement.h"

// DDCore dependencies
#include "DetectorDescription/DDBase/interface/DDdebug.h"
#include "DetectorDescription/DDBase/interface/DDException.h"
#include "DetectorDescription/DDCore/interface/DDConstant.h"
#include "DetectorDescription/DDCore/interface/DDCurrentNamespace.h"

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
  string* sp = new string("*** root ***");
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
  
  char * buf = XMLString::transcode(qname);
  map<string, string*>::const_iterator namePtr = namesMap_.find(string(buf));
  string* nameInt;
  if (namePtr != namesMap_.end())
    {
      nameInt = namePtr->second;
    }
  else
    {
      string * sp = new string(buf);
      nameInt = sp;
      namesMap_[*sp] = nameInt;
    }
  names_.push_back(nameInt);
  string myElementName = string(buf);

  //  cout << "start: namesMap_ = " << *(names_.back()) << "  names_.back()= " << names_.back() << endl;

  delete[] buf;

//    char * xmlc = XMLString::transcode(qname);
//    string myElementName = string(xmlc);
//    delete[] xmlc;
  //DDLSAX2Handler::startElement(uri, localname, qname, attrs);
  //  cout << "names_.top() = " << names_.top() << endl;
  // count types of elements.  Note, this is cumulative for all files handled by
  // this handler.
  ++elementTypeCounter_[myElementName];
  DDXMLElement* myElement = DDLElementRegistry::getElement(myElementName);

  unsigned int numAtts = attrs.getLength();
  vector<string> attrNames, attrValues;

  for (unsigned int i = 0; i < numAtts; i++)
    {
      char * buf = XMLString::transcode(attrs.getLocalName(i));
      string myattname(buf); delete[] buf;
      buf = XMLString::transcode(attrs.getValue(i));
      string myvalue(buf); delete[] buf;
      buf = XMLString::transcode(attrs.getQName(i));
      string myQName(buf); delete[] buf;
      attrNames.push_back(myattname);
      attrValues.push_back(myvalue);
    }

  if (myElement != NULL)
    {
      DDLParser* beingParsed = DDLParser::Instance();
      string nmspace = getnmspace(extractFileName( beingParsed->getCurrFileName()));
      myElement->loadAttributes(myElementName, attrNames, attrValues, nmspace);
      //  initialize text
      myElement->loadText(string()); 
    }

  else 
    {
      string s = "DDLSAX2FileHandler::startElement ERROR No pointer returned from";
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
  string myElementName = *(names_.back());

  DCOUT_V('P', "DDLSAX2FileHandler::endElement started");
  DCOUT_V('P', "    " + myElementName);

  DDXMLElement* myElement = DDLElementRegistry::getElement(myElementName);
  DDLParser* beingParsed = DDLParser::Instance();
  if (myElement != NULL)
    {
      string nmspace = getnmspace(extractFileName( beingParsed->getCurrFileName()));
      DDCurrentNamespace::ns() = nmspace;
      try {
	myElement->processElement(myElementName, nmspace);
      }
      catch (DDException & e) {
        string s(e.what());
	s+="\nDDLSAX2FileHandler::endElement call to processElement Failed.";
	myElement->throwError(s);
      }
    }
  else 
    {
      string s = "DDLSAX2FileHandler::endElement ERROR No pointer returned from";
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
//   string myElementName = "**defaulted**";
//   if (names_.size() == 1)
//     {
//       myElement = new DDXMLElement;
//       myElement->loadText(string());
//     }
//   else
//     {
      myElement = DDLElementRegistry::getElement(*(names_.back()));
      //      string myElementName = *(names_.back());
//     }

  string instring = "";
  for (unsigned int i = 0; i < length; i++)
    {
      char s = chars[i];
      instring = instring + s;
    }
  if (myElement->gotText())
    myElement->appendText(instring);
  else
    myElement->loadText(instring);

  DCOUT_V('P', "DDLSAX2FileHandler::characters completed"); 
}

void DDLSAX2FileHandler::comment( const   XMLCh* const    chars
				  , const unsigned int    length)
{
  // ignore, discard, overkill since base class also has this...
}

string DDLSAX2FileHandler::extractFileName(string fullname)
{
  string ret = "";
  size_t startfrom = fullname.size() - 1;
  while (startfrom > 0 && fullname[startfrom] != '/')
    startfrom--;
  if (fullname[startfrom] == '/') startfrom = startfrom + 1;
  ret = fullname.substr(startfrom, fullname.size() - startfrom);
  return ret;
}

void DDLSAX2FileHandler::dumpElementTypeCounter()
{
  for (map<string, int>::const_iterator it = elementTypeCounter_.begin();
       it != elementTypeCounter_.end(); it++)
    cout << "Element: " << it->first << " (" << it->second << ")" << endl;
}

void DDLSAX2FileHandler::createDDConstants() const
{
  try {
    DDConstant::createConstantsFromEvaluator();
  }
  catch(DDException & e) {
    string msg ("caught in DDLSAX2FileHandler::createDDConstants():\n");
    msg = msg + e.what();
    throw DDException(msg);
  }
}

string& DDLSAX2FileHandler::parent() 
{
  if (names_.size() > 1)
    {
      return *(names_[names_.size() - 2]);
    }
  return *(names_[0]);
}

string& DDLSAX2FileHandler::self()
{
  if (names_.size() > 2)
    return *(names_[names_.size() - 1]);
  return *(names_[0]);
}
