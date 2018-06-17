#ifndef DETECTOR_DESCRIPTION_PARSER_DDL_SAX2_FILE_HANDLER_H
#define DETECTOR_DESCRIPTION_PARSER_DDL_SAX2_FILE_HANDLER_H

#include <cstddef>
#include <map>
#include <string>
#include <vector>
#include <xercesc/sax2/Attributes.hpp>

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"
#include "DetectorDescription/Parser/interface/DDLSAX2Handler.h"

class DDCompactView;
class DDLElementRegistry;

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
  
  DDLSAX2FileHandler( DDCompactView& cpv, DDLElementRegistry& );
  ~DDLSAX2FileHandler() override;

  void init() ;

  // -----------------------------------------------------------------------
  //  Handlers for the SAX ContentHandler interface
  // -----------------------------------------------------------------------
  
  void startElement( const XMLCh* uri, const XMLCh* localname,
		     const XMLCh* qname, const Attributes& attrs) override;
  void endElement( const XMLCh* uri, const XMLCh* localname,
		   const XMLCh* qname) override;
  void characters( const XMLCh* chars, XMLSize_t length) override;
  void comment( const XMLCh* chars, XMLSize_t length ) override;

  //! creates all DDConstant from the evaluator which has been already 'filled' in the first scan of the documents
  void createDDConstants() const; 

 protected:
  DDLElementRegistry& registry() { return registry_; }
 private:
  virtual const std::string& parent() const;
  virtual const std::string& self() const;
  
 private:

  std::vector< std::string > namesMap_;
  std::vector< size_t > names_;
  DDCompactView& cpv_;
  DDLElementRegistry& registry_;
};

#endif
