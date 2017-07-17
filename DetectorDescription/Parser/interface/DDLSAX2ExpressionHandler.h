#ifndef DETECTORDESCRIPTION_PARSER_DDLSAX2EXPRESSIONHANDLER_H
#define DETECTORDESCRIPTION_PARSER_DDLSAX2EXPRESSIONHANDLER_H

#include <xercesc/sax2/Attributes.hpp>
#include <string>

#include "DetectorDescription/Parser/interface/DDLSAX2FileHandler.h"
#include "DetectorDescription/Parser/interface/DDLSAX2Handler.h"

class DDCompactView;

/// DDLSAX2ExpressionHandler is the first pass SAX2 Handler for XML files found in the configuration file.
/** @class DDLSAX2ExpressionHandler
 * @author Michael Case
 *
 *  DDLSAX2ExpressionHandler.h  -  description
 *  -------------------
 *  begin: Mon Feb 25, 2002
 * 
 *  This processes only ConstantsSection/Parameter elements so there is no need
 *  to make it as elaborate as the second pass parser.
 *
 */
class DDLSAX2ExpressionHandler : public DDLSAX2FileHandler 
{
 public:

  DDLSAX2ExpressionHandler(DDCompactView& cpv);
  ~DDLSAX2ExpressionHandler() override;

  void startElement(const XMLCh* const uri, const XMLCh* const localname,
		    const XMLCh* const qname, const Attributes& attrs) override;
  
  void endElement(const XMLCh* const uri, const XMLCh* const localname,
		  const XMLCh* const qname) override;
};

#endif
