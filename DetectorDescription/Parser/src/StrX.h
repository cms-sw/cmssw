#ifndef DETECTOR_DESCRIPTION_PARSER_STRX_H
#define DETECTOR_DESCRIPTION_PARSER_STRX_H

#include <xercesc/util/XercesDefs.hpp>
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

  StrX(const XMLCh* const toTranscode)
    {
      fLocalForm = XMLString::transcode(toTranscode);
    }

  ~StrX()
    {
      XMLString::release(&fLocalForm);
    }

  const char* localForm() const
    {
      return fLocalForm;
    }

private:
  char * fLocalForm;
};

inline std::ostream& operator<<(std::ostream& target, const StrX& toDump)
{
  target << toDump.localForm();
  return target;
}

#endif
