// file      : xsd/cxx/xml/char-lcp.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#include <cstring> // std::memcpy

#include <xercesc/util/XMLString.hpp>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/auto-array.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/std-memory-manager.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace xml
    {
      template <typename C>
      std::basic_string<C> char_lcp_transcoder<C>::
      to (const XMLCh* s)
      {
        std_memory_manager mm;
        auto_array<C, std_memory_manager> r (
          xercesc::XMLString::transcode (s, &mm), mm);
        return std::basic_string<C> (r.get ());
      }

      template <typename C>
      std::basic_string<C> char_lcp_transcoder<C>::
      to (const XMLCh* s, std::size_t len)
      {
        auto_array<XMLCh> tmp (new XMLCh[len + 1]);
        std::memcpy (tmp.get (), s, len * sizeof (XMLCh));
        tmp[len] = XMLCh (0);

        std_memory_manager mm;
        auto_array<C, std_memory_manager> r (
          xercesc::XMLString::transcode (tmp.get (), &mm), mm);

        tmp.reset ();

        return std::basic_string<C> (r.get ());
      }

      template <typename C>
      XMLCh* char_lcp_transcoder<C>::
      from (const C* s)
      {
        std_memory_manager mm;
        return xercesc::XMLString::transcode (s, &mm);
      }
    }
  }
}
