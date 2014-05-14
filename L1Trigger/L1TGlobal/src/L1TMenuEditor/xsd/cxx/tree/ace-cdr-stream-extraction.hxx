// file      : xsd/cxx/tree/ace-cdr-stream-extraction.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_TREE_ACE_CDR_STREAM_EXTRACTION_HXX
#define XSD_CXX_TREE_ACE_CDR_STREAM_EXTRACTION_HXX

#include <cstddef> // std::size_t
#include <string>

#include <ace/ACE.h> // ACE::strdelete
#include <ace/CDR_Stream.h>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/auto-array.hxx>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/buffer.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/istream.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/ace-cdr-stream-common.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      struct ace_cdr_stream_extraction: ace_cdr_stream_operation
      {
        virtual const char*
        what () const throw ()
        {
          return "ACE CDR stream extraction operation failed";
        }
      };


      // as_size
      //

#ifdef XSD_CXX_TREE_USE_64_BIT_SIZE
      template <typename T>
      inline istream<ACE_InputCDR>&
      operator>> (istream<ACE_InputCDR>& s,
                  istream<ACE_InputCDR>::as_size<T>& x)
      {
        ACE_CDR::ULongLong r;

        if (!s.impl ().read_ulonglong (r) ||
            r > ~(T (0)))
          throw ace_cdr_stream_extraction ();

        x.x_ = static_cast<T> (r);

        return s;
      }
#else
      template <typename T>
      inline istream<ACE_InputCDR>&
      operator>> (istream<ACE_InputCDR>& s,
                  istream<ACE_InputCDR>::as_size<T>& x)
      {
        ACE_CDR::ULong r;

        if (!s.impl ().read_ulong (r))
          throw ace_cdr_stream_extraction ();

        x.x_ = static_cast<T> (r);

        return s;
      }
#endif


      // 8-bit
      //
      template <typename T>
      inline istream<ACE_InputCDR>&
      operator>> (istream<ACE_InputCDR>& s,
                  istream<ACE_InputCDR>::as_int8<T>& x)
      {
        ACE_CDR::Octet r;

        if (!s.impl ().read_octet (r))
          throw ace_cdr_stream_extraction ();

        x.x_ = static_cast<T> (r);

        return s;
      }

      template <typename T>
      inline istream<ACE_InputCDR>&
      operator>> (istream<ACE_InputCDR>& s,
                  istream<ACE_InputCDR>::as_uint8<T>& x)
      {
        ACE_CDR::Octet r;

        if (!s.impl ().read_octet (r))
          throw ace_cdr_stream_extraction ();

        x.x_ = static_cast<T> (r);

        return s;
      }


      // 16-bit
      //
      template <typename T>
      inline istream<ACE_InputCDR>&
      operator>> (istream<ACE_InputCDR>& s,
                  istream<ACE_InputCDR>::as_int16<T>& x)
      {
        ACE_CDR::Short r;

        if (!s.impl ().read_short (r))
          throw ace_cdr_stream_extraction ();

        x.x_ = static_cast<T> (r);

        return s;
      }

      template <typename T>
      inline istream<ACE_InputCDR>&
      operator>> (istream<ACE_InputCDR>& s,
                  istream<ACE_InputCDR>::as_uint16<T>& x)
      {
        ACE_CDR::UShort r;

        if (!s.impl ().read_ushort (r))
          throw ace_cdr_stream_extraction ();

        x.x_ = static_cast<T> (r);

        return s;
      }


      // 32-bit
      //
      template <typename T>
      inline istream<ACE_InputCDR>&
      operator>> (istream<ACE_InputCDR>& s,
                  istream<ACE_InputCDR>::as_int32<T>& x)
      {
        ACE_CDR::Long r;

        if (!s.impl ().read_long (r))
          throw ace_cdr_stream_extraction ();

        x.x_ = static_cast<T> (r);

        return s;
      }

      template <typename T>
      inline istream<ACE_InputCDR>&
      operator>> (istream<ACE_InputCDR>& s,
                  istream<ACE_InputCDR>::as_uint32<T>& x)
      {
        ACE_CDR::ULong r;

        if (!s.impl ().read_ulong (r))
          throw ace_cdr_stream_extraction ();

        x.x_ = static_cast<T> (r);

        return s;
      }


      // 64-bit
      //
      template <typename T>
      inline istream<ACE_InputCDR>&
      operator>> (istream<ACE_InputCDR>& s,
                  istream<ACE_InputCDR>::as_int64<T>& x)
      {
        ACE_CDR::LongLong r;

        if (!s.impl ().read_longlong (r))
          throw ace_cdr_stream_extraction ();

        x.x_ = static_cast<T> (r);

        return s;
      }

      template <typename T>
      inline istream<ACE_InputCDR>&
      operator>> (istream<ACE_InputCDR>& s,
                  istream<ACE_InputCDR>::as_uint64<T>& x)
      {
        ACE_CDR::ULongLong r;

        if (!s.impl ().read_ulonglong (r))
          throw ace_cdr_stream_extraction ();

        x.x_ = static_cast<T> (r);

        return s;
      }


      // Boolean
      //
      template <typename T>
      inline istream<ACE_InputCDR>&
      operator>> (istream<ACE_InputCDR>& s,
                  istream<ACE_InputCDR>::as_bool<T>& x)
      {
        ACE_CDR::Boolean r;

        if (!s.impl ().read_boolean (r))
          throw ace_cdr_stream_extraction ();

        x.x_ = static_cast<T> (r);

        return s;
      }


      // Floating-point
      //
      template <typename T>
      inline istream<ACE_InputCDR>&
      operator>> (istream<ACE_InputCDR>& s,
                  istream<ACE_InputCDR>::as_float32<T>& x)
      {
        ACE_CDR::Float r;

        if (!s.impl ().read_float (r))
          throw ace_cdr_stream_extraction ();

        x.x_ = static_cast<T> (r);

        return s;
      }

      template <typename T>
      inline istream<ACE_InputCDR>&
      operator>> (istream<ACE_InputCDR>& s,
                  istream<ACE_InputCDR>::as_float64<T>& x)
      {
        ACE_CDR::Double r;

        if (!s.impl ().read_double (r))
          throw ace_cdr_stream_extraction ();

        x.x_ = static_cast<T> (r);

        return s;
      }

      // Extraction of std::basic_string.
      //

      namespace bits
      {
        template<typename C>
        struct ace_str_deallocator
        {
          void
          deallocate (C* s)
          {
            ACE::strdelete (s);
          }
        };
      }

      inline istream<ACE_InputCDR>&
      operator>> (istream<ACE_InputCDR>& s, std::basic_string<char>& x)
      {
        typedef bits::ace_str_deallocator<char> deallocator;

        deallocator d;
        char* r;

        if (!s.impl ().read_string (r))
          throw ace_cdr_stream_extraction ();

        auto_array<char, deallocator> ar (r, d);

        x = r;

        return s;
      }

#ifdef ACE_HAS_WCHAR
      inline istream<ACE_InputCDR>&
      operator>> (istream<ACE_InputCDR>& s, std::basic_string<wchar_t>& x)
      {
        typedef bits::ace_str_deallocator<wchar_t> deallocator;

        deallocator d;
        wchar_t* r;

        if (!s.impl ().read_wstring (r))
          throw ace_cdr_stream_extraction ();

        auto_array<wchar_t, deallocator> ar (r, d);

        x = r;

        return s;
      }
#endif


      // Extraction of a binary buffer.
      //
      template <typename C>
      istream<ACE_InputCDR>&
      operator>> (istream<ACE_InputCDR>& s, buffer<C>& x)
      {
        ACE_CDR::ULong size;

        if (!s.impl ().read_ulong (size))
          throw ace_cdr_stream_extraction ();

        x.size (size);

        if (!s.impl ().read_octet_array (
              reinterpret_cast<ACE_CDR::Octet*> (x.data ()), size))
          throw ace_cdr_stream_extraction ();

        return s;
      }
    }
  }
}

#endif  // XSD_CXX_TREE_ACE_CDR_STREAM_EXTRACTION_HXX
