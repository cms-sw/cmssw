// file      : xsd/cxx/xml/string.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_XML_STRING_HXX
#define XSD_CXX_XML_STRING_HXX

#include <string>
#include <cstddef> // std::size_t

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/auto-array.hxx>
#include <xercesc/util/XercesDefs.hpp> // XMLCh

namespace xsd
{
  namespace cxx
  {
    namespace xml
    {
      // Transcode a null-terminated string.
      //
      template <typename C>
      std::basic_string<C>
      transcode (const XMLCh* s);

      // Transcode a potentially non-null-terminated string.
      //
      template <typename C>
      std::basic_string<C>
      transcode (const XMLCh* s, std::size_t length);


      // For VC7.1 wchar_t and XMLCh are the same type so we cannot
      // overload the transcode name. You should not use these functions
      // anyway and instead use the xml::string class below.
      //
      template <typename C>
      XMLCh*
      transcode_to_xmlch (const C*);

      template <typename C>
      XMLCh*
      transcode_to_xmlch (const std::basic_string<C>& s);

      //
      //
      class string
      {
      public :
        template <typename C>
        string (const std::basic_string<C>& s)
            : s_ (transcode_to_xmlch<C> (s))
        {
        }

        template <typename C>
        string (const C* s)
            : s_ (transcode_to_xmlch<C> (s))
        {
        }

        const XMLCh*
        c_str () const
        {
          return s_.get ();
        }

      private:
        string (const string&);

        string&
        operator= (const string&);

      private:
        auto_array<XMLCh> s_;
      };
    }
  }
}

#endif // XSD_CXX_XML_STRING_HXX

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/string.ixx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/string.txx>
