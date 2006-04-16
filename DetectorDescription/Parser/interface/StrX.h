/***************************************************************************
                          StrX.h  -  description
                             -------------------
    begin                : Tue Oct 23 2001
    copyright            : See Xerces C++ documentation
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/

#ifndef STRX_H
#define STRX_H

#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/sax2/XMLReaderFactory.hpp>
#include <xercesc/sax2/SAX2XMLReader.hpp>
#include <xercesc/util/XMLString.hpp>
#include <string>
#include <iostream>

namespace xercesc_2_3 { } using namespace xercesc_2_3;

/** @class StrX
 * @author Apache Xerces C++ Example
 *
 *           DDDParser sub-component of DDD
 *
 *  This is taken from the Examples of Apache Xerces C++ and used 'as is'.
 *
 */
// ---------------------------------------------------------------------------
//  This is a simple class that lets us do easy (though not terribly efficient)
//  trancoding of XMLCh data to local code page for display.
// ---------------------------------------------------------------------------
class StrX
{
  public :
    // -----------------------------------------------------------------------
    //  Constructors and Destructor
    // -----------------------------------------------------------------------
    StrX(const XMLCh* const toTranscode)// : fXMLChForm(toTranscode)
    {
      char * fLocalForm;
      fLocalForm = XMLString::transcode(toTranscode);
      fStoredForm = std::string (fLocalForm);
      delete [] fLocalForm;
    }

  StrX( const char* const toTranscode )// : fLocalForm(toTranscode) {
    {
      XMLCh * fXMLChForm;
      fXMLChForm = XMLString::transcode(toTranscode);
      fStoredForm = std::string(toTranscode);
      delete [] fXMLChForm;
    }

  StrX( const std::string& toTranscode ) : fStoredForm(toTranscode)
    {

    }

  ~StrX()
    {

    }

  // -----------------------------------------------------------------------
  //  Getter methods
  // -----------------------------------------------------------------------
  const char* localForm() const
    {
      return fStoredForm.c_str();
    }

  const XMLCh* xmlChForm() const
    {
      return XMLString::transcode(fStoredForm.c_str());
    }

  const std::string& stringForm() const {
    return fStoredForm;
  }

  private :
    // -----------------------------------------------------------------------
    //  Private data members
    //
    //  fLocalForm
    //      This is the local code page form of the std::string.
    // -----------------------------------------------------------------------
    std::string fStoredForm;
};

inline std::ostream& operator<<(std::ostream& target, const StrX& toDump)
{
  target << toDump.localForm();
  return target;
}
#endif
