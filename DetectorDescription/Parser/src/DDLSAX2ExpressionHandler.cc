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

// ---------------------------------------------------------------------------
//  Includes
// ---------------------------------------------------------------------------
#include "DetectorDescription/Parser/interface/DDLSAX2ExpressionHandler.h"
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Parser/interface/StrX.h"

// DDCore dependencies
#include "DetectorDescription/Base/interface/DDdebug.h"

#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

// Xerces C++ dependencies
#include <xercesc/sax2/Attributes.hpp>
#include <xercesc/sax/SAXParseException.hpp>
#include <xercesc/sax/SAXException.hpp>

// COBRA timing
//#include "Utilities/Notification/interface/TimerProxy.h"
//#include "Utilities/Notification/interface/TimingReport.h"

#include "SealUtil/SealTimer.h"

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>

// ---------------------------------------------------------------------------
//  DDLSAX2Handler: Constructors and Destructor
// ---------------------------------------------------------------------------
DDLSAX2ExpressionHandler::DDLSAX2ExpressionHandler() // : inConstantsSection_(false)
{
}

DDLSAX2ExpressionHandler::~DDLSAX2ExpressionHandler()
{
}

// This does most of the work, it sets and determines whether it is 
// in a ConstantsSection element, and if so, to process all the 
// constants accordingly.
void DDLSAX2ExpressionHandler::startElement(const XMLCh* const uri
                                   , const XMLCh* const localname
                                   , const XMLCh* const qname
                                   , const Attributes& attrs)
{
  static seal::SealTimer tdseh("DDLSAX2ExpressionHandler::startElement(..)", false);
  
  ++elementCount_;
  attrCount_ += attrs.getLength();

  //char * tmpc = XMLString::transcode(qname);
  pElementName = StrX(qname).stringForm();//std::string(tmpc);
  //  delete[] tmpc; tmpc=0;

  if (pElementName == "Constant") // && pInConstantsSection)
    {
      ++elementTypeCounter_[pElementName];
      DCOUT_V('P', std::string("DDLSAX2ExpressionHandler: start ") + pElementName);
      unsigned int numAtts = attrs.getLength();
      std::string varName, varValue;
      for (unsigned int i = 0; i < numAtts; ++i)
	{
	  std::string myattname = StrX(attrs.getLocalName(i)).stringForm();
	  std::string myvalue = StrX(attrs.getValue(i)).stringForm();

	  std::string myQName = StrX(attrs.getQName(i)).stringForm();
	  DCOUT_V('P', std::string("DDLSAX2ExpressionHandler: ") + "getLocalName = " + myattname + "  getValue = " +  myvalue + "   getQName = " + myQName);

	  // attributes unit and quantity are not used right now.
	  if (myattname == "name")
	    varName = myvalue;
	  else if (myattname == "value")
	    varValue = myvalue;
	}
      DDLParser* beingParsed = DDLParser::instance();
      std::string nmspace = getnmspace(extractFileName( beingParsed->getCurrFileName()));
      ExprEvalInterface & ev = ExprEvalSingleton::instance();
      ev.set(nmspace, varName, varValue);
    }
}

void DDLSAX2ExpressionHandler::endElement(const XMLCh* const uri
				    , const XMLCh* const localname
				    , const XMLCh* const qname)
{

}

