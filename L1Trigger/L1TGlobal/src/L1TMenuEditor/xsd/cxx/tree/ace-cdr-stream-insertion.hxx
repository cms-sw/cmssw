// file      : xsd/cxx/tree/ace-cdr-stream-insertion.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_TREE_ACE_CDR_STREAM_INSERTION_HXX
#define XSD_CXX_TREE_ACE_CDR_STREAM_INSERTION_HXX

#include <cstddef> // std::size_t
#include <string>

#include <ace/CDR_Stream.h>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/buffer.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/ostream.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/ace-cdr-stream-common.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      struct ace_cdr_stream_insertion: ace_cdr_stream_operation
      {
        virtual const char*
        what () const throw ()
        {
          return "ACE CDR stream insertion operation failed";
        }
      };


      // as_size
      //

#ifdef XSD_CXX_TREE_USE_64_BIT_SIZE
      template <typename T>
      inline ostream<ACE_OutputCDR>&
      operator<< (ostream<ACE_OutputCDR>& s,
                  ostream<ACE_OutputCDR>::as_size<T> x)
      {
        if (!s.impl ().write_ulonglong (
              static_cast<ACE_CDR::ULongLong> (x.x_)))
          throw ace_cdr_stream_insertion ();
        return s;
      }
#else
      template <typename T>
      inline ostream<ACE_OutputCDR>&
      operator<< (ostream<ACE_OutputCDR>& s,
                  ostream<ACE_OutputCDR>::as_size<T> x)
      {
        if (x.x_ > ~(ACE_CDR::ULong (0)) ||
            !s.impl ().write_ulong (static_cast<ACE_CDR::ULong> (x.x_)))
          throw ace_cdr_stream_insertion ();

        return s;
      }
#endif


      // 8-bit
      //
      template <typename T>
      inline ostream<ACE_OutputCDR>&
      operator<< (ostream<ACE_OutputCDR>& s,
                  ostream<ACE_OutputCDR>::as_int8<T> x)
      {
        ACE_CDR::Octet r (static_cast<ACE_CDR::Octet> (x.x_));

        if (!s.impl ().write_octet (r))
          throw ace_cdr_stream_insertion ();

        return s;
      }

      template <typename T>
      inline ostream<ACE_OutputCDR>&
      operator<< (ostream<ACE_OutputCDR>& s,
                  ostream<ACE_OutputCDR>::as_uint8<T> x)
      {
        ACE_CDR::Octet r (static_cast<ACE_CDR::Octet> (x.x_));

        if (!s.impl ().write_octet (r))
          throw ace_cdr_stream_insertion ();

        return s;
      }


      // 16-bit
      //
      template <typename T>
      inline ostream<ACE_OutputCDR>&
      operator<< (ostream<ACE_OutputCDR>& s,
                  ostream<ACE_OutputCDR>::as_int16<T> x)
      {
        if (!s.impl ().write_short (static_cast<ACE_CDR::Short> (x.x_)))
          throw ace_cdr_stream_insertion ();
        return s;
      }

      template <typename T>
      inline ostream<ACE_OutputCDR>&
      operator<< (ostream<ACE_OutputCDR>& s,
                  ostream<ACE_OutputCDR>::as_uint16<T> x)
      {
        if (!s.impl ().write_ushort (static_cast<ACE_CDR::UShort> (x.x_)))
          throw ace_cdr_stream_insertion ();
        return s;
      }


      // 32-bit
      //
      template <typename T>
      inline ostream<ACE_OutputCDR>&
      operator<< (ostream<ACE_OutputCDR>& s,
                  ostream<ACE_OutputCDR>::as_int32<T> x)
      {
        if (!s.impl ().write_long (static_cast<ACE_CDR::Long> (x.x_)))
          throw ace_cdr_stream_insertion ();
        return s;
      }

      template <typename T>
      inline ostream<ACE_OutputCDR>&
      operator<< (ostream<ACE_OutputCDR>& s,
                  ostream<ACE_OutputCDR>::as_uint32<T> x)
      {
        if (!s.impl ().write_ulong (static_cast<ACE_CDR::ULong> (x.x_)))
          throw ace_cdr_stream_insertion ();
        return s;
      }


      // 64-bit
      //
      template <typename T>
      inline ostream<ACE_OutputCDR>&
      operator<< (ostream<ACE_OutputCDR>& s,
                  ostream<ACE_OutputCDR>::as_int64<T> x)
      {
        if (!s.impl ().write_longlong (static_cast<ACE_CDR::LongLong> (x.x_)))
          throw ace_cdr_stream_insertion ();
        return s;
      }

      template <typename T>
      inline ostream<ACE_OutputCDR>&
      operator<< (ostream<ACE_OutputCDR>& s,
                  ostream<ACE_OutputCDR>::as_uint64<T> x)
      {
        if (!s.impl ().write_ulonglong (
              static_cast<ACE_CDR::ULongLong> (x.x_)))
          throw ace_cdr_stream_insertion ();
        return s;
      }


      // Boolean
      //
      template <typename T>
      inline ostream<ACE_OutputCDR>&
      operator<< (ostream<ACE_OutputCDR>& s,
                  ostream<ACE_OutputCDR>::as_bool<T> x)
      {
        if (!s.impl ().write_boolean (static_cast<ACE_CDR::Boolean> (x.x_)))
          throw ace_cdr_stream_insertion ();
        return s;
      }


      // Floating-point
      //
      template <typename T>
      inline ostream<ACE_OutputCDR>&
      operator<< (ostream<ACE_OutputCDR>& s,
                  ostream<ACE_OutputCDR>::as_float32<T> x)
      {
        if (!s.impl ().write_float (static_cast<ACE_CDR::Float> (x.x_)))
          throw ace_cdr_stream_insertion ();
        return s;
      }

      template <typename T>
      inline ostream<ACE_OutputCDR>&
      operator<< (ostream<ACE_OutputCDR>& s,
                  ostream<ACE_OutputCDR>::as_float64<T> x)
      {
        if (!s.impl ().write_double (static_cast<ACE_CDR::Double> (x.x_)))
          throw ace_cdr_stream_insertion ();
        return s;
      }

      // Insertion of std::basic_string.
      //

      inline ostream<ACE_OutputCDR>&
      operator<< (ostream<ACE_OutputCDR>& s, const std::basic_string<char>& x)
      {
        // ACE CDR strings are hard-wired with a 32 bit length.
        //
        if (x.length () > ~(ACE_CDR::ULong (0)) ||
            !s.impl ().write_string (
              static_cast<ACE_CDR::ULong> (x.length ()), x.c_str ()))
          throw ace_cdr_stream_insertion ();
        return s;
      }

#ifdef ACE_HAS_WCHAR
      inline ostream<ACE_OutputCDR>&
      operator<< (ostream<ACE_OutputCDR>& s,
                  const std::basic_string<wchar_t>& x)
      {
        // ACE CDR strings are hard-wired with a 32 bit length.
        //
        if (x.length () > ~(ACE_CDR::ULong (0)) ||
            !s.impl ().write_wstring (
              static_cast<ACE_CDR::ULong> (x.length ()), x.c_str ()))
          throw ace_cdr_stream_insertion ();
        return s;
      }
#endif

      // Insertion of a binary buffer.
      //
      template <typename C>
      ostream<ACE_OutputCDR>&
      operator<< (ostream<ACE_OutputCDR>& s, const buffer<C>& x)
      {
        std::size_t size (x.size ());

        // It is not possible to write an array with a 64-bit size.
        //
        if (size > ~(ACE_CDR::ULong (0)) ||
            !s.impl ().write_ulong (static_cast<ACE_CDR::ULong> (size)) ||
            !s.impl ().write_octet_array (
              reinterpret_cast<const ACE_CDR::Octet*> (x.data ()), size))
          throw ace_cdr_stream_insertion ();

        return s;
      }
    }
  }
}

#endif  // XSD_CXX_TREE_ACE_CDR_STREAM_INSERTION_HXX
