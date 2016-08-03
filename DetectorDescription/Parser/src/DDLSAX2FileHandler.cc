#include "DetectorDescription/Parser/interface/DDLSAX2FileHandler.h"
#include "DetectorDescription/Core/interface/DDConstant.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Parser/src/DDXMLElement.h"
#include "Utilities/Xerces/interface/XercesStrUtils.h"

using namespace cms::xerces;

class DDCompactView;

// XERCES_CPP_NAMESPACE_USE 

DDLSAX2FileHandler::DDLSAX2FileHandler( DDCompactView & cpv )
  : cpv_(cpv)
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

void
DDLSAX2FileHandler::startElement( const XMLCh* const uri,
				  const XMLCh* const localname,
				  const XMLCh* const qname,
				  const Attributes& attrs )
{
  std::string myElementName(cStr(qname).ptr());
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

  DDXMLElement* myElement = DDLGlobalRegistry::instance().getElement(myElementName);

  unsigned int numAtts = attrs.getLength();
  std::vector<std::string> attrNames, attrValues;

  for (unsigned int i = 0; i < numAtts; ++i)
  {
    attrNames.push_back(std::string(cStr(attrs.getLocalName(i)).ptr()));
    attrValues.push_back(std::string(cStr(attrs.getValue(i)).ptr()));
  }
  
  myElement->loadAttributes(myElementName, attrNames, attrValues, nmspace_, cpv_);
  //  initialize text
  myElement->loadText(std::string()); 
}

void
DDLSAX2FileHandler::endElement( const XMLCh* const uri,
				const XMLCh* const localname,
				const XMLCh* const qname )
{
  std::string ts(cStr(qname).ptr());
  const std::string&  myElementName = self();

  DDXMLElement* myElement = DDLGlobalRegistry::instance().getElement(myElementName);

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

  DDCurrentNamespace::ns() = nmspace;
  // tell the element it's parent name for recording/reporting purposes
  myElement->setParent(parent());
  myElement->setSelf(self());
  myElement->processElement(myElementName, nmspace, cpv_);

  names_.pop_back();
}

void
DDLSAX2FileHandler::characters( const XMLCh* const chars,
				const XMLSize_t length )
{
  DDXMLElement* myElement = DDLGlobalRegistry::instance().getElement(self());
  std::string inString = "";
  for (XMLSize_t i = 0; i < length; ++i)
  {
    char s = chars[i];
    inString = inString + s;
  }
  if (myElement->gotText())
    myElement->appendText(inString);
  else
    myElement->loadText(inString);
}

void
DDLSAX2FileHandler::comment( const XMLCh* const chars,
			     const XMLSize_t length )
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
