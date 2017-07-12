#ifndef DETECTORDESCRIPTION_PARSER_DDLSAX2HANDLER_H
#define DETECTORDESCRIPTION_PARSER_DDLSAX2HANDLER_H

#include <xercesc/sax2/Attributes.hpp>
#include <xercesc/sax2/DefaultHandler.hpp>
#include <iostream>
#include <string>
#include <vector>

///  DDLSAX2Handler inherits from Xerces C++ DefaultHandler.
/** @class DDLSAX2Handler
 * @author Michael Case
 *
 *  DDLSAX2Handler.h  -  description
 *  -------------------
 *  begin: Mon Oct 22 2001
 *
 *  The DefaultHandler of Xerces C++ provides an interface to the SAX2 event
 *  driven processing of XML documents.  It does so by providing methods which
 *  are redefined by the inheriting class (DDLSAX2Handler in this case) to
 *  provide the desired processing for each event.
 *
 *  In this case, we accumulate some statistics.  This class does nothing with
 *  startElement and endElement events.
 *
 */
class DDLSAX2Handler : public XERCES_CPP_NAMESPACE::DefaultHandler
{
 public:
  typedef XERCES_CPP_NAMESPACE::Attributes Attributes;
  typedef XERCES_CPP_NAMESPACE::SAXParseException SAXParseException;

  DDLSAX2Handler();
  ~DDLSAX2Handler() override;

  /// Get the count of elements processed so far.
  unsigned int getElementCount() const
  {
    return elementCount_;
  }
  /// Get the count of attributes processed so far.
  unsigned int getAttrCount() const
  {
    return attrCount_;
  }
  /// Get the count of characters processed so far.
  unsigned int getCharacterCount() const
  {
    return characterCount_;
  }
  /// Did the XML parser see any errors?
  bool getSawErrors() const
  {
    return sawErrors_;
  }
  /// Get the count of spaces processed so far.
  unsigned int getSpaceCount() const
  {
    return spaceCount_;
  }

  // -----------------------------------------------------------------------
  //  Handlers for the SAX ContentHandler interface
  // -----------------------------------------------------------------------

  void startElement(const XMLCh* const uri, const XMLCh* const localname,
			    const XMLCh* const qname, const Attributes& attrs) override;
  void endElement(const XMLCh* const uri, const XMLCh* const localname,
			  const XMLCh* const qname) override;
  void characters(const XMLCh* const chars, const XMLSize_t length) override;
  void comment (const XMLCh *const chars, const XMLSize_t length ) override;
  void ignorableWhitespace(const XMLCh* const chars, const XMLSize_t length) override;
  void resetDocument() override;

  // -----------------------------------------------------------------------
  //  Handlers for the SAX ErrorHandler interface
  // -----------------------------------------------------------------------
  void warning(const SAXParseException& exception) override;
  void error(const SAXParseException& exception) override;
  void fatalError(const SAXParseException& exception) override;
  virtual void dumpStats(const std::string& fname);
  
 protected:

  // -----------------------------------------------------------------------
  //  Protected data members
  //
  //  attrCount_
  //  characterCount_
  //  elementCount_
  //  spaceCount_
  //      These are just counters that are run upwards based on the input
  //      from the document handlers.
  //
  //  sawErrors
  //      This is set by the error handlers, and is queryable later to
  //      see if any errors occurred.
  // -----------------------------------------------------------------------
  XMLSize_t       attrCount_;
  XMLSize_t       characterCount_;
  XMLSize_t       elementCount_;
  XMLSize_t       spaceCount_;
  bool            sawErrors_;
  bool            userNS_;
  std::string     nmspace_;
 
 public:
  /** This allows the DDLSAX2Handler and objects that inherit from it to set
   ** the userNS_ flag to indicate 
   **     false[default] use the filename of the file being handled as the DD namespace
   **     true           assume ALL the "name" attributes have DD namespace specified.
   **/
  virtual void setUserNS(bool userns);
  virtual void setNameSpace(const std::string& nms);
};

#endif
