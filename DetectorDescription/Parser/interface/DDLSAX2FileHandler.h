#ifndef DETECTORDESCRIPTION_PARSER_DDL_SAX2FILEHANDLER_H
#define DETECTORDESCRIPTION_PARSER_DDL_SAX2FILEHANDLER_H

#include <stddef.h>
#include <xercesc/sax2/Attributes.hpp>
#include <map>
#include <string>
#include <vector>

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/interface/DDLSAX2Handler.h"

class DDCompactView;

/// DDLSAX2FileHandler is the SAX2 Handler for XML files found in the configuration file.
/** @class DDLSAX2FileHandler
 * @author Michael Case
 *
 *  DDLSAX2FileHandler.h  -  description
 *  -------------------
 *  begin: Tue Oct 23 2001
 *
 *  DDLSAX2FileHandler has the same structure as the DDLSAX2ConfigHandler as they
 *  both inherit from DDLSAX2Handler which inherits from Xerces C++ DefaultHandler.
 *  SAX2 is event driven.  So, when the start of an element is encountered in the 
 *  XML, then a call is made to the handler's startElement.  The same for endElement.
 * 
 *  The design of DDXMLElement allows for processing whichever type of Element the
 *  XML Parser encounters.
 *
 */
class DDLSAX2FileHandler : public DDLSAX2Handler 
{
 public:
  
  DDLSAX2FileHandler( DDCompactView& cpv );
  ~DDLSAX2FileHandler() override;

  void init() ;

  // -----------------------------------------------------------------------
  //  Handlers for the SAX ContentHandler interface
  // -----------------------------------------------------------------------
  
  void startElement(const XMLCh* const uri, const XMLCh* const localname,
		    const XMLCh* const qname, const Attributes& attrs) override;
  void endElement(const XMLCh* const uri, const XMLCh* const localname,
		  const XMLCh* const qname) override;
  void characters (const XMLCh *const chars, const XMLSize_t length) override;
  void comment (const XMLCh *const chars, const XMLSize_t length ) override;
  
 private:
  virtual const std::string& parent() const;
  virtual const std::string& self() const;
  
 private:
  //! creates all DDConstant from the evaluator which has been already 'filled' in the first scan of the documents
  void createDDConstants() const; 

  std::vector< std::string > namesMap_;
  std::vector< size_t > names_;
  DDCompactView& cpv_;
};

#endif
