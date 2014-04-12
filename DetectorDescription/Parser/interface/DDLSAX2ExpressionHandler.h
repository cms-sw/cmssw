#ifndef DDL_SAX2ExpressionHandler_H
#define DDL_SAX2ExpressionHandler_H

// ---------------------------------------------------------------------------
//  Includes
// ---------------------------------------------------------------------------
// Parser parts.
#include "DetectorDescription/Parser/interface/DDLSAX2FileHandler.h"

// Xerces dependencies
#include <xercesc/sax2/Attributes.hpp>

/// DDLSAX2ExpressionHandler is the first pass SAX2 Handler for XML files found in the configuration file.
/** @class DDLSAX2ExpressionHandler
 * @author Michael Case
 *
 *  DDLSAX2ExpressionHandler.h  -  description
 *  -------------------
 *  begin: Mon Feb 25, 2002
 *  email: case@ucdhep.ucdavis.edu
 * 
 *  This processes only ConstantsSection/Parameter elements so there is no need
 *  to make it as elaborate as the second pass parser.
 *
 */
class DDLSAX2ExpressionHandler : public DDLSAX2FileHandler 
{

 public:

    // -----------------------------------------------------------------------
    //  Constructor and Destructor
    // -----------------------------------------------------------------------

    DDLSAX2ExpressionHandler(DDCompactView& cpv);
    ~DDLSAX2ExpressionHandler();

    // -----------------------------------------------------------------------
    //  Handlers for the SAX ContentHandler interface
    // -----------------------------------------------------------------------

    void startElement(const XMLCh* const uri, const XMLCh* const localname
		      , const XMLCh* const qname, const Attributes& attrs);
    void endElement(const XMLCh* const uri, const XMLCh* const localname
		    , const XMLCh* const qname);

   private: 
    std::string pElementName;

};

#endif
