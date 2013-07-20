#ifndef XERCES_STRINGS_H
#define XERCES_STRINGS_H

#include <string>
#include <boost/scoped_array.hpp>
#include <xercesc/util/XMLString.hpp>
#include <iostream>


/** Utility functions to convert from unhandy XMLCh * to std::string and back

    To convert a XMLCh* into a std::string:
 
    std::string aString= toNative(const XMLCh* str);

    to convert a std::string into XMLCh* and not worry about memory:
    
    XMLCh * aCh = fromNative(std::string str).c_str();
    
    \author Stefano Argiro' (seen somehwere on the internet)
    $Id: XercesString.h,v 1.1 2008/11/14 15:46:05 argiro Exp $

 */

namespace xuti {
  
  /// Define an intermediate type
  typedef std::basic_string<XMLCh> XercesString;
  
  // Converts from a narrow-character string to a wide-character string.
  inline XercesString fromNative(const char* str){
    boost::scoped_array<XMLCh> ptr(xercesc::XMLString::transcode(str));
    return XercesString(ptr.get( ));
  }
  
  // Converts from a narrow-character string to a wide-charactr string.
  inline XercesString fromNative(const std::string& str){
    return fromNative(str.c_str( ));
  }
  
  // Converts from a wide-character string to a narrow-character string.
  inline std::string toNative(const XMLCh* str){
    boost::scoped_array<char> ptr(xercesc::XMLString::transcode(str));
    return std::string(ptr.get( ));
  }
  
  // Converts from a wide-character string to a narrow-character string.
  inline std::string toNative(const XercesString& str){
    return toNative(str.c_str( ));
  }


} // namespace 

#endif // #ifndef XERCES_STRINGS_H
