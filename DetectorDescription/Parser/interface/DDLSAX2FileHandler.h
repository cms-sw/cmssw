#ifndef DDL_SAX2FileHandler_H
#define DDL_SAX2FileHandler_H

// ---------------------------------------------------------------------------
//  Includes
// ---------------------------------------------------------------------------
// Parser parts.
#include "DetectorDescription/Parser/interface/DDLSAX2Handler.h"
#include "DetectorDescription/Parser/interface/DDLElementRegistry.h"

// DDCore parts
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

// Xerces dependencies
#include <xercesc/sax2/Attributes.hpp>

#include <string>
#include <vector>
#include <map>

/// DDLSAX2FileHandler is the SAX2 Handler for XML files found in the configuration file.
/** @class DDLSAX2FileHandler
 * @author Michael Case
 *
 *  DDLSAX2FileHandler.h  -  description
 *  -------------------
 *  begin: Tue Oct 23 2001
 *  email: case@ucdhep.ucdavis.edu
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
  
  // -----------------------------------------------------------------------
  //  Constructor and Destructor
  // -----------------------------------------------------------------------
  
  //  DDLSAX2FileHandler();
  DDLSAX2FileHandler( DDCompactView& cpv );
  ~DDLSAX2FileHandler();

  void init() ;
  //  void setStorage( DDCompactView & cpv );

  // -----------------------------------------------------------------------
  //  Handlers for the SAX ContentHandler interface
  // -----------------------------------------------------------------------
  
  void startElement(const XMLCh* const uri, const XMLCh* const localname
		    , const XMLCh* const qname, const Attributes& attrs);
  void endElement(const XMLCh* const uri, const XMLCh* const localname
		  , const XMLCh* const qname);
  void characters (const XMLCh *const chars, const unsigned int length);
  void comment (const XMLCh *const chars, const unsigned int length );
  
  //  virtual std::string extractFileName(std::string fullname);
  
  virtual const std::string& parent() const;
  virtual const std::string& self() const;
  
  // -----------------------------------------------------------------------
  //  Dump information on number and name of elements processed.
  // -----------------------------------------------------------------------
  /// This dumps some statistics on elements encountered in the file.
  void dumpElementTypeCounter();

 protected:
  //! creates all DDConstant from the evaluator which has been already 'filled' in the first scan of the documents
  void createDDConstants() const; 
  //  Map that holds name and number of elements processed.
  std::map < std::string, int> elementTypeCounter_;
  std::vector<std::string> namesMap_;
  std::vector < size_t > names_;
  DDCompactView& cpv_;
  DDLElementRegistry xmlelems_;
};

#endif
