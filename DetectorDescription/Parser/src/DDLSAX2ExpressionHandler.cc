/***************************************************************************
                          DDLSAX2ExpressionHandler.cc  -  description
                             -------------------
    begin                : Mon Feb 25, 2002
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *           DDDParser sub-component of DDD                                *
 *                                                                         *
 ***************************************************************************/

#include "DetectorDescription/Parser/interface/DDLSAX2ExpressionHandler.h"
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Parser/src/StrX.h"

#include "DetectorDescription/Base/interface/DDdebug.h"

#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

#include <iostream>
#include <vector>
#include <string>

// ---------------------------------------------------------------------------
//  DDLSAX2Handler: Constructors and Destructor
// ---------------------------------------------------------------------------
DDLSAX2ExpressionHandler::DDLSAX2ExpressionHandler( DDCompactView& cpv )
  : DDLSAX2FileHandler::DDLSAX2FileHandler( cpv )
{}

DDLSAX2ExpressionHandler::~DDLSAX2ExpressionHandler( void )
{}

// This does most of the work, it sets and determines whether it is 
// in a ConstantsSection element, and if so, to process all the 
// constants accordingly.
void
DDLSAX2ExpressionHandler::startElement( const XMLCh* const uri,
					const XMLCh* const localname,
					const XMLCh* const qname,
					const Attributes& attrs )
{
  ++elementCount_;
  attrCount_ += attrs.getLength();

  pElementName = StrX(qname).localForm();

  if (pElementName == "Constant") 
  {
    ++elementTypeCounter_[pElementName];
    DCOUT_V('P', std::string("DetectorDescription/Parser/interface/DDLSAX2ExpressionHandler: start ") + pElementName);
    unsigned int numAtts = attrs.getLength();
    std::string varName, varValue;
    for (unsigned int i = 0; i < numAtts; ++i)
    {
      std::string myattname(StrX(attrs.getLocalName(i)).localForm());
      std::string myvalue(StrX(attrs.getValue(i)).localForm());

      std::string myQName(StrX(attrs.getQName(i)).localForm());
      DCOUT_V('P', std::string("DetectorDescription/Parser/interface/DDLSAX2ExpressionHandler: ") + "getLocalName = " + myattname + "  getValue = " +  myvalue + "   getQName = " + myQName);

      // attributes unit and quantity are not used right now.
      if (myattname == "name")
	varName = myvalue;
      else if (myattname == "value")
	varValue = myvalue;
    }
    //      DDLParser* beingParsed = DDLParser::instance();
    //      std::string nmspace = getnmspace(extractFileName( beingParsed->getCurrFileName()));
    ExprEvalInterface & ev = ExprEvalSingleton::instance();
    ev.set(nmspace_, varName, varValue);
  }
}

void
DDLSAX2ExpressionHandler::endElement( const XMLCh* const uri,
				      const XMLCh* const localname,
				      const XMLCh* const qname )
{}
