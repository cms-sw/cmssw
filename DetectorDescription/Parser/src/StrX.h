/***************************************************************************
                          StrX.h  -  description
                             -------------------
    begin                : Tue Oct 23 2001
    copyright            : See Xerces C++ documentation
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/

#ifndef STRX_H
#define STRX_H

#include <xercesc/util/XercesDefs.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/sax2/XMLReaderFactory.hpp>
#include <xercesc/sax2/SAX2XMLReader.hpp>
#include <xercesc/util/XMLString.hpp>
#include <string>
#include <iostream>

/** @class StrX
 * @author Apache Xerces C++ Example
 *
 *           DDDParser sub-component of DDD
 *
 *  This is taken from the Examples of Apache Xerces C++ and modified.
 *
 */
// ---------------------------------------------------------------------------
//  This is a simple class that lets us do easy (though not terribly efficient)
//  trancoding of XMLCh data to local code page for display.
// ---------------------------------------------------------------------------
class StrX
{
public:
  typedef XERCES_CPP_NAMESPACE::XMLString XMLString;
  // -----------------------------------------------------------------------
  //  Constructors and Destructor
  // -----------------------------------------------------------------------
  StrX(const XMLCh* const toTranscode)// : fXMLChForm(toTranscode)
    {
      fLocalForm = XMLString::transcode(toTranscode);
      fXMLChForm = XMLString::transcode(fLocalForm);
    }

  StrX( const char* const toTranscode )
    {
      fXMLChForm = XMLString::transcode(toTranscode);
      fLocalForm = XMLString::transcode(fXMLChForm);
    }

  StrX( const std::string& toTranscode )
    {
      fXMLChForm = XMLString::transcode(toTranscode.c_str());
      fLocalForm = XMLString::transcode(fXMLChForm);
    }
  
  ~StrX()
    {
      XMLString::release(&fLocalForm);
      XMLString::release(&fXMLChForm);
    }

  // -----------------------------------------------------------------------
  //  Getter methods
  // -----------------------------------------------------------------------
  const char* localForm() const
    {
      return fLocalForm;
    }

  const XMLCh* xmlChForm() const
    {
      return fXMLChForm;
    }

private:
  XMLCh * fXMLChForm;
  char * fLocalForm;
  
};

inline std::ostream& operator<<(std::ostream& target, const StrX& toDump)
{
  target << toDump.localForm();
  return target;
}
#endif
