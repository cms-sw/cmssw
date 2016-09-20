#ifndef DETECTOR_DESCRIPTION_PARSER_XERCEC_STRING_H
# define DETECTOR_DESCRIPTION_PARSER_XERCEC_STRING_H

#include <xercesc/util/XercesDefs.hpp>
#include <xercesc/util/XMLString.hpp>
#include <memory>

#ifdef XERCES_CPP_NAMESPACE_USE
XERCES_CPP_NAMESPACE_USE
#endif

class XercesString
{
 public:
  XercesString() : _wstr(nullptr) { };
  XercesString(const char *str);
  XercesString(XMLCh *wstr);
  XercesString(const XMLCh *wstr);
  XercesString(const XercesString &copy);
  ~XercesString();
  bool append(const XMLCh *tail);
  bool erase(const XMLCh *head, const XMLCh *tail);
  const XMLCh* begin() const;
  const XMLCh* end() const;
  int size() const;
  XMLCh & operator [] (const int i);
  const XMLCh operator [] (const int i) const;
  operator const XMLCh * () const { return _wstr; };
 private:
  XMLCh* _wstr;
};

#endif 
