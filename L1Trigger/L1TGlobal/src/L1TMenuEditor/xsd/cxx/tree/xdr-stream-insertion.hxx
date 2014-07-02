// file      : xsd/cxx/tree/xdr-stream-insertion.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_TREE_XDR_STREAM_INSERTION_HXX
#define XSD_CXX_TREE_XDR_STREAM_INSERTION_HXX

#include <rpc/types.h>
#include <rpc/xdr.h>

#include <string>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/buffer.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/ostream.hxx>
#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/xdr-stream-common.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      struct xdr_stream_insertion: xdr_stream_operation
      {
        virtual const char*
        what () const throw ()
        {
          return "XDR stream insertion operation failed";
        }
      };

      // as_size
      //
#ifdef XSD_CXX_TREE_USE_64_BIT_SIZE
      template <typename T>
      inline ostream<XDR>&
      operator<< (ostream<XDR>& s, ostream<XDR>::as_size<T> x)
      {
        uint64_t v (static_cast<uint64_t> (x.x_));

        if (!xdr_uint64_t (&s.impl (), &v))
          throw xdr_stream_insertion ();

        return s;
      }
#else
      template <typename T>
      inline ostream<XDR>&
      operator<< (ostream<XDR>& s, ostream<XDR>::as_size<T> x)
      {
        uint32_t v (static_cast<uint32_t> (x.x_));

        if (x.x_ > ~(uint32_t (0)) || !xdr_uint32_t (&s.impl (), &v))
          throw xdr_stream_insertion ();

        return s;
      }
#endif


      // 8-bit
      //
      template <typename T>
      inline ostream<XDR>&
      operator<< (ostream<XDR>& s, ostream<XDR>::as_int8<T> x)
      {
        int8_t v (static_cast<int8_t> (x.x_));

        if (!xdr_int8_t (&s.impl (), &v))
          throw xdr_stream_insertion ();

        return s;
      }

      template <typename T>
      inline ostream<XDR>&
      operator<< (ostream<XDR>& s, ostream<XDR>::as_uint8<T> x)
      {
        uint8_t v (static_cast<uint8_t> (x.x_));

        if (!xdr_uint8_t (&s.impl (), &v))
          throw xdr_stream_insertion ();

        return s;
      }


      // 16-bit
      //
      template <typename T>
      inline ostream<XDR>&
      operator<< (ostream<XDR>& s, ostream<XDR>::as_int16<T> x)
      {
        int16_t v (static_cast<int16_t> (x.x_));

        if (!xdr_int16_t (&s.impl (), &v))
          throw xdr_stream_insertion ();

        return s;
      }

      template <typename T>
      inline ostream<XDR>&
      operator<< (ostream<XDR>& s, ostream<XDR>::as_uint16<T> x)
      {
        uint16_t v (static_cast<uint16_t> (x.x_));

        if (!xdr_uint16_t (&s.impl (), &v))
          throw xdr_stream_insertion ();

        return s;
      }


      // 32-bit
      //
      template <typename T>
      inline ostream<XDR>&
      operator<< (ostream<XDR>& s, ostream<XDR>::as_int32<T> x)
      {
        int32_t v (static_cast<int32_t> (x.x_));

        if (!xdr_int32_t (&s.impl (), &v))
          throw xdr_stream_insertion ();

        return s;
      }

      template <typename T>
      inline ostream<XDR>&
      operator<< (ostream<XDR>& s, ostream<XDR>::as_uint32<T> x)
      {
        uint32_t v (static_cast<uint32_t> (x.x_));

        if (!xdr_uint32_t (&s.impl (), &v))
          throw xdr_stream_insertion ();

        return s;
      }


      // 64-bit
      //
      template <typename T>
      inline ostream<XDR>&
      operator<< (ostream<XDR>& s, ostream<XDR>::as_int64<T> x)
      {
        int64_t v (static_cast<int64_t> (x.x_));

        if (!xdr_int64_t (&s.impl (), &v))
          throw xdr_stream_insertion ();

        return s;
      }

      template <typename T>
      inline ostream<XDR>&
      operator<< (ostream<XDR>& s, ostream<XDR>::as_uint64<T> x)
      {
        uint64_t v (static_cast<uint64_t> (x.x_));

        if (!xdr_uint64_t (&s.impl (), &v))
          throw xdr_stream_insertion ();

        return s;
      }


      // Boolean
      //
      template <typename T>
      inline ostream<XDR>&
      operator<< (ostream<XDR>& s, ostream<XDR>::as_bool<T> x)
      {
        bool_t v (static_cast<bool_t> (x.x_));

        if (!xdr_bool (&s.impl (), &v))
          throw xdr_stream_insertion ();

        return s;
      }


      // Floating-point
      //
      template <typename T>
      inline ostream<XDR>&
      operator<< (ostream<XDR>& s, ostream<XDR>::as_float32<T> x)
      {
        float v (static_cast<float> (x.x_));

        if (!xdr_float (&s.impl (), &v))
          throw xdr_stream_insertion ();

        return s;
      }

      template <typename T>
      inline ostream<XDR>&
      operator<< (ostream<XDR>& s, ostream<XDR>::as_float64<T> x)
      {
        double v (static_cast<double> (x.x_));

        if (!xdr_double (&s.impl (), &v))
          throw xdr_stream_insertion ();

        return s;
      }

      // Insertion of std::basic_string.
      //

      inline ostream<XDR>&
      operator<< (ostream<XDR>& s, const std::basic_string<char>& x)
      {
        // XDR strings are hard-wired with a 32 bit (unsigned int) length.
        //
        char* p (const_cast<char*> (x.c_str ()));
        unsigned int n (static_cast<unsigned int> (x.length ()));

        if (x.length () > ~((unsigned int) 0) ||
            !xdr_u_int (&s.impl (), &n) ||
            !xdr_opaque (&s.impl (), p, n))
          throw xdr_stream_insertion ();

        return s;
      }

      // Wide strings are not supported by XDR.
      //
      // inline ostream<XDR>&
      // operator<< (ostream<XDR>& s, const std::basic_string<wchar_t>& x)
      // {
      // }


      // Insertion of a binary buffer.
      //
      template <typename C>
      ostream<XDR>&
      operator<< (ostream<XDR>& s, const buffer<C>& x)
      {
        // It is not possible to write an array with a 64-bit size.
        //
        unsigned int n (static_cast<unsigned int> (x.size ()));

        if (x.size () > ~((unsigned int) 0) ||
            !xdr_u_int (&s.impl (), &n) ||
            !xdr_opaque (&s.impl (), const_cast<char*> (x.data ()), n))
          throw xdr_stream_insertion ();

        return s;
      }
    }
  }
}

#endif  // XSD_CXX_TREE_XDR_STREAM_INSERTION_HXX
