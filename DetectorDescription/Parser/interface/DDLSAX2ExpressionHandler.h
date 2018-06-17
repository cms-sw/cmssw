#ifndef DETECTOR_DESCRIPTION_PARSER_DDL_SAX2_EXPRESSION_HANDLER_H
#define DETECTOR_DESCRIPTION_PARSER_DDL_SAX2_EXPRESSION_HANDLER_H

#include <xercesc/sax2/Attributes.hpp>
#include <string>

#include "DetectorDescription/Parser/interface/DDLSAX2FileHandler.h"
#include "DetectorDescription/Parser/interface/DDLSAX2Handler.h"

class DDCompactView;
class DDLElementRegistry;

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

  DDLSAX2ExpressionHandler(DDCompactView& cpv, DDLElementRegistry&);
  ~DDLSAX2ExpressionHandler() override;

  void startElement( const XMLCh* uri, const XMLCh* localname,
		     const XMLCh* qname, const Attributes& attrs) override;
  
  void endElement( const XMLCh* uri, const XMLCh* localname,
		   const XMLCh* qname) override;
};

#endif
