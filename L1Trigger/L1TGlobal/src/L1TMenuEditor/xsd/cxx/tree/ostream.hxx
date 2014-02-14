// file      : xsd/cxx/tree/ostream.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_TREE_OSTREAM_HXX
#define XSD_CXX_TREE_OSTREAM_HXX

#include <cstddef> // std::size_t

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      class ostream_common
      {
      public:
        template <typename T>
        struct as_size
        {
          explicit as_size (T x) : x_ (x) {}
          T x_;
        };


        // 8-bit
        //
        template <typename T>
        struct as_int8
        {
          explicit as_int8 (T x) : x_ (x) {}
          T x_;
        };

        template <typename T>
        struct as_uint8
        {
          explicit as_uint8 (T x) : x_ (x) {}
          T x_;
        };


        // 16-bit
        //
        template <typename T>
        struct as_int16
        {
          explicit as_int16 (T x) : x_ (x) {}
          T x_;
        };

        template <typename T>
        struct as_uint16
        {
          explicit as_uint16 (T x) : x_ (x) {}
          T x_;
        };


        // 32-bit
        //
        template <typename T>
        struct as_int32
        {
          explicit as_int32 (T x) : x_ (x) {}
          T x_;
        };

        template <typename T>
        struct as_uint32
        {
          explicit as_uint32 (T x) : x_ (x) {}
          T x_;
        };


        // 64-bit
        //
        template <typename T>
        struct as_int64
        {
          explicit as_int64 (T x) : x_ (x) {}
          T x_;
        };

        template <typename T>
        struct as_uint64
        {
          explicit as_uint64 (T x) : x_ (x) {}
          T x_;
        };


        // Boolean
        //
        template <typename T>
        struct as_bool
        {
          explicit as_bool (T x) : x_ (x) {}
          T x_;
        };


        // Floating-point
        //
        template <typename T>
        struct as_float32
        {
          explicit as_float32 (T x) : x_ (x) {}
          T x_;
        };

        template <typename T>
        struct as_float64
        {
          explicit as_float64 (T x) : x_ (x) {}
          T x_;
        };
      };

      template<typename S>
      class ostream: public ostream_common
      {
      public:
        explicit
        ostream (S& s)
            : s_ (s)
        {
        }

        S&
        impl ()
        {
          return s_;
        }

      private:
        ostream (const ostream&);
        ostream&
        operator= (const ostream&);

      private:
        S& s_;
      };


      // 8-bit
      //
      template <typename S>
      inline ostream<S>&
      operator<< (ostream<S>& s, signed char x)
      {
        return s << ostream_common::as_int8<signed char> (x);
      }

      template <typename S>
      inline ostream<S>&
      operator<< (ostream<S>& s, unsigned char x)
      {
        return s << ostream_common::as_uint8<unsigned char> (x);
      }


      // 16-bit
      //
      template <typename S>
      inline ostream<S>&
      operator<< (ostream<S>& s, short x)
      {
        return s << ostream_common::as_int16<short> (x);
      }

      template <typename S>
      inline ostream<S>&
      operator<< (ostream<S>& s, unsigned short x)
      {
        return s << ostream_common::as_uint16<unsigned short> (x);
      }


      // 32-bit
      //
      template <typename S>
      inline ostream<S>&
      operator<< (ostream<S>& s, int x)
      {
        return s << ostream_common::as_int32<int> (x);
      }

      template <typename S>
      inline ostream<S>&
      operator<< (ostream<S>& s, unsigned int x)
      {
        return s << ostream_common::as_uint32<unsigned int> (x);
      }


      // 64-bit
      //
      template <typename S>
      inline ostream<S>&
      operator<< (ostream<S>& s, long long x)
      {
        return s << ostream_common::as_int64<long long> (x);
      }

      template <typename S>
      inline ostream<S>&
      operator<< (ostream<S>& s, unsigned long long x)
      {
        return s << ostream_common::as_uint64<unsigned long long> (x);
      }

      // Boolean
      //
      template <typename S>
      inline ostream<S>&
      operator<< (ostream<S>& s, bool x)
      {
        return s << ostream_common::as_bool<bool> (x);
      }


      // Floating-point
      //
      template <typename S>
      inline ostream<S>&
      operator<< (ostream<S>& s, float x)
      {
        return s << ostream_common::as_float32<float> (x);
      }

      template <typename S>
      inline ostream<S>&
      operator<< (ostream<S>& s, double x)
      {
        return s << ostream_common::as_float64<double> (x);
      }
    }
  }
}

#endif  // XSD_CXX_TREE_OSTREAM_HXX
