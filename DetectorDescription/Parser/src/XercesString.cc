#include "DetectorDescription/Parser/interface/XercesString.h"

#include <stddef.h>
#include <stdlib.h>
#include <iostream>

#ifdef XERCES_CPP_NAMESPACE_USE
XERCES_CPP_NAMESPACE_USE
#endif

XercesString::XercesString( const char *str )
  : _wstr( nullptr )
{
  _wstr = XMLString::transcode( str );
}

XercesString::XercesString( XMLCh *wstr )
  : _wstr( wstr )
{}

XercesString::XercesString( const XMLCh *wstr )
  : _wstr( nullptr )
{
  _wstr = XMLString::replicate( wstr );
}

XercesString::XercesString( const XercesString &right )
  : _wstr( nullptr )
{
  _wstr = XMLString::replicate( right._wstr );
}

XercesString::~XercesString()
{
  if( _wstr ) XMLString::release( &_wstr );
}

bool
XercesString::append( const XMLCh *tail )
{
  int iTailLen = XMLString::stringLen(tail);
  int iWorkLen = XMLString::stringLen(_wstr);
  XMLCh *result = new XMLCh[ iWorkLen + iTailLen + 1 ];
  bool bOK = result != nullptr;
  if( bOK )
  {
    XMLCh *target = result;
    XMLString::moveChars(target, _wstr, iWorkLen);
    target += iWorkLen;
    XMLString::moveChars(target, tail, iTailLen);
    target += iTailLen;
    *target++ = 0;
    XMLString::release(&_wstr);
    _wstr = result;
  }
  return bOK;
}

bool
XercesString::erase(const XMLCh *head, const XMLCh *tail)
{
  bool bOK = head <= tail && head >= begin() && tail <= end();
  if (bOK)
  {
    XMLCh *result = new XMLCh[ size() - (tail - head) + 1 ];
    XMLCh *target = result;
    bOK = target != nullptr;
    if (bOK)
    {
      const XMLCh *cursor = begin();

      while (cursor != head) *target++ = *cursor++;
      cursor = tail;
      while ( cursor != end() ) *target++ = *cursor++;
      *target ++ = 0;
      XMLString::release(&_wstr);
      _wstr = result;
    }
  }
  return bOK;
}

const XMLCh* XercesString::begin() const
{
  return _wstr;
}

const XMLCh* XercesString::end() const
{
  return _wstr + size();
}

int XercesString::size() const
{
  return XMLString::stringLen(_wstr);
}

XMLCh & XercesString::operator [] (const int i)
{
  return _wstr[i];
}

const XMLCh XercesString::operator [] (const int i) const
{
  return _wstr[i];
}
