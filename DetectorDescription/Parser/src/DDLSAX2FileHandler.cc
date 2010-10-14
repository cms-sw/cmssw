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

#include "DetectorDescription/Parser/interface/DDLSAX2FileHandler.h"
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Parser/src/StrX.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"

#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Core/interface/DDConstant.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"

#include <iostream>

// XERCES_CPP_NAMESPACE_USE 

DDLSAX2FileHandler::DDLSAX2FileHandler( DDCompactView & cpv )
  : cpv_(cpv),
    xmlelems_()
{
  init();
}

void
DDLSAX2FileHandler::init( void )
{
  createDDConstants();
  namesMap_.push_back("*** root ***");
  names_.push_back(namesMap_.size() - 1);
}

DDLSAX2FileHandler::~DDLSAX2FileHandler( void )
{}

// ---------------------------------------------------------------------------
//  DDLSAX2Handler: Implementation of the SAX DocumentHandler interface
// ---------------------------------------------------------------------------
void
DDLSAX2FileHandler::startElement(const XMLCh* const uri
				 , const XMLCh* const localname
				 , const XMLCh* const qname
				 , const Attributes& attrs)
{
  DCOUT_V('P', "DDLSAX2FileHandler::startElement started");

  std::string myElementName(StrX(qname).localForm());
  size_t i = 0;
  for (; i < namesMap_.size(); ++i) {
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
  //final way
  //  DDXMLElement* myElement = xmlelems_.getElement(myElementName); //myRegistry_->getElement(myElementName);
  //temporary way:
  DDXMLElement* myElement = DDLGlobalRegistry::instance().getElement(myElementName);

  unsigned int numAtts = attrs.getLength();
  std::vector<std::string> attrNames, attrValues;

  for (unsigned int i = 0; i < numAtts; ++i)
  {
    //       char* temp2 = XMLString::transcode(attrs.getLocalName(i));
    //       char* temp3 = XMLString::transcode(attrs.getValue(i));
    attrNames.push_back(std::string(StrX(attrs.getLocalName(i)).localForm()));
    attrValues.push_back(std::string(StrX(attrs.getValue(i)).localForm()));
    //       XMLString::release(&temp2);
    //       XMLString::release(&temp3);
  }
  
  myElement->loadAttributes(myElementName, attrNames, attrValues, nmspace_, cpv_);
  //  initialize text
  myElement->loadText(std::string()); 
  DCOUT_V('P', "DDLSAX2FileHandler::startElement completed");
}

void
DDLSAX2FileHandler::endElement( const XMLCh* const uri,
				const XMLCh* const localname,
				const XMLCh* const qname )
{
  std::string ts(StrX(qname).localForm());
  const std::string&  myElementName = self();
  DCOUT_V('P', "DDLSAX2FileHandler::endElement started");
  DCOUT_V('P', "    " + myElementName);
  //final way
  //  DDXMLElement* myElement = xmlelems_.getElement(myElementName); //myRegistry_->getElement(myElementName);
  //temporary way:
  DDXMLElement* myElement = DDLGlobalRegistry::instance().getElement(myElementName);

  //   DDLParser* beingParsed = DDLParser::instance();
  //   std::string nmspace = getnmspace(extractFileName( beingParsed->getCurrFileName()));
  std::string nmspace = nmspace_;
  // The need for processElement to have the nmspace so that it can 
  // do the necessary gymnastics made things more complicated in the
  // effort to allow fully user-controlled namespaces.  So the "magic"
  // trick of setting nmspace to "!" is used :(... I don't like this magic trick
  // -- Michael Case 2008-11-06
  // OPTIMISE in the near future, like the current nmspace_ impl. 
  // just set nmspace_ to "!" from Parser based on userNS_ so 
  // userNS_ is set by parser ONCE and no if nec. here. MEC: 2009-06-22
  if ( userNS_ ) {
    nmspace = "!";
  } 
  //  std::cout << "namespace " << nmspace_ << std::endl;
  DDCurrentNamespace::ns() = nmspace;
  // tell the element it's parent name for recording/reporting purposes
  myElement->setParent(parent());
  myElement->setSelf(self());
  myElement->processElement(myElementName, nmspace, cpv_);
  DCOUT_V('P', "DDLSAX2FileHandler::endElement completed");
  names_.pop_back();
}

void
DDLSAX2FileHandler::characters( const XMLCh* const chars,
				const unsigned int length )
{
  DCOUT_V('P', "DDLSAX2FileHandler::characters started");
  //  std::cout << "character handler started" << std::endl;
  //DDXMLElement* myElement = NULL;
  // final way
  //  myElement = xmlelems_.getElement(namesMap_[names_.back()]);
  //temporary way:
  //  const std::string&  myElementName = namesMap_[names_.back()];
  DDXMLElement* myElement = DDLGlobalRegistry::instance().getElement(self());//myElementName); //namesMap_[names_.back()]);
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

void
DDLSAX2FileHandler::comment( const   XMLCh* const    chars
			     , const unsigned int    length)
{
  // ignore, discard, overkill since base class also has this...
}

void
DDLSAX2FileHandler::dumpElementTypeCounter( void )
{}

void
DDLSAX2FileHandler::createDDConstants( void ) const
{
  DDConstant::createConstantsFromEvaluator();
}

const std::string&
DDLSAX2FileHandler::parent( void ) const
{
  if (names_.size() > 2)
  {
    return namesMap_.at(names_.at(names_.size() - 2));
  }
  return namesMap_[0];//.at(names_.at(0));
}

const std::string&
DDLSAX2FileHandler::self( void ) const
{
  if (names_.size() > 1) {
    return namesMap_.at(names_.at(names_.size() - 1));
  }
  return namesMap_[0];//.at(names_.at(0));
}
