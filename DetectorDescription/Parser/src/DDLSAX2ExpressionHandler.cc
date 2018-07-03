#include "DetectorDescription/Parser/interface/DDLSAX2ExpressionHandler.h"
#include "DetectorDescription/Core/interface/ClhepEvaluator.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "Utilities/Xerces/interface/XercesStrUtils.h"

#include <map>
#include <string>

using namespace cms::xerces;

class DDCompactView;

DDLSAX2ExpressionHandler::DDLSAX2ExpressionHandler( DDCompactView& cpv, DDLElementRegistry& reg )
  : DDLSAX2FileHandler::DDLSAX2FileHandler( cpv,reg )
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
  if( XMLString::equals( qname, uStr("Constant").ptr())) 
  {
    std::string varName = toString( attrs.getValue( uStr( "name" ).ptr()));
    std::string varValue = toString( attrs.getValue( uStr( "value" ).ptr()));
    ClhepEvaluator & ev = registry().evaluator();
    ev.set(nmspace_, varName, varValue);
  }
}

void
DDLSAX2ExpressionHandler::endElement( const XMLCh* const uri,
				      const XMLCh* const localname,
				      const XMLCh* const qname )
{}
