// file      : xsd/cxx/tree/istream.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_TREE_ISTREAM_HXX
#define XSD_CXX_TREE_ISTREAM_HXX

#include <cstddef> // std::size_t

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/istream-fwd.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      class istream_common
      {
      public:
        template <typename T>
        struct as_size
        {
          explicit as_size (T& x) : x_ (x) {}
          T& x_;
        };


        // 8-bit
        //
        template <typename T>
        struct as_int8
        {
          explicit as_int8 (T& x) : x_ (x) {}
          T& x_;
        };

        template <typename T>
        struct as_uint8
        {
          explicit as_uint8 (T& x) : x_ (x) {}
          T& x_;
        };


        // 16-bit
        //
        template <typename T>
        struct as_int16
        {
          explicit as_int16 (T& x) : x_ (x) {}
          T& x_;
        };

        template <typename T>
        struct as_uint16
        {
          explicit as_uint16 (T& x) : x_ (x) {}
          T& x_;
        };


        // 32-bit
        //
        template <typename T>
        struct as_int32
        {
          explicit as_int32 (T& x) : x_ (x) {}
          T& x_;
        };

        template <typename T>
        struct as_uint32
        {
          explicit as_uint32 (T& x) : x_ (x) {}
          T& x_;
        };


        // 64-bit
        //
        template <typename T>
        struct as_int64
        {
          explicit as_int64 (T& x) : x_ (x) {}
          T& x_;
        };

        template <typename T>
        struct as_uint64
        {
          explicit as_uint64 (T& x) : x_ (x) {}
          T& x_;
        };


        // Boolean
        //
        template <typename T>
        struct as_bool
        {
          explicit as_bool (T& x) : x_ (x) {}
          T& x_;
        };


        // Floating-point
        //
        template <typename T>
        struct as_float32
        {
          explicit as_float32 (T& x) : x_ (x) {}
          T& x_;
        };

        template <typename T>
        struct as_float64
        {
          explicit as_float64 (T& x) : x_ (x) {}
          T& x_;
        };
      };

      template<typename S>
      class istream: public istream_common
      {
      public:
        explicit
        istream (S& s)
            : s_ (s)
        {
        }

        S&
        impl ()
        {
          return s_;
        }

      private:
        istream (const istream&);
        istream&
        operator= (const istream&);

      private:
        S& s_;
      };


      // 8-bit
      //
      template <typename S>
      inline istream<S>&
      operator>> (istream<S>& s, signed char& x)
      {
        istream_common::as_int8<signed char> as_int8 (x);
        return s >> as_int8;
      }

      template <typename S>
      inline istream<S>&
      operator>> (istream<S>& s, unsigned char& x)
      {
        istream_common::as_uint8<unsigned char> as_uint8 (x);
        return s >> as_uint8;
      }


      // 16-bit
      //
      template <typename S>
      inline istream<S>&
      operator>> (istream<S>& s, short& x)
      {
        istream_common::as_int16<short> as_int16 (x);
        return s >> as_int16;
      }

      template <typename S>
      inline istream<S>&
      operator>> (istream<S>& s, unsigned short& x)
      {
        istream_common::as_uint16<unsigned short> as_uint16 (x);
        return s >> as_uint16;
      }


      // 32-bit
      //
      template <typename S>
      inline istream<S>&
      operator>> (istream<S>& s, int& x)
      {
        istream_common::as_int32<int> as_int32 (x);
        return s >> as_int32;
      }

      template <typename S>
      inline istream<S>&
      operator>> (istream<S>& s, unsigned int& x)
      {
        istream_common::as_uint32<unsigned int> as_uint32 (x);
        return s >> as_uint32;
      }


      // 64-bit
      //
      template <typename S>
      inline istream<S>&
      operator>> (istream<S>& s, long long& x)
      {
        istream_common::as_int64<long long> as_int64 (x);
        return s >> as_int64;
      }

      template <typename S>
      inline istream<S>&
      operator>> (istream<S>& s, unsigned long long& x)
      {
        istream_common::as_uint64<unsigned long long> as_uint64 (x);
        return s >> as_uint64;
      }

      // Boolean
      //
      template <typename S>
      inline istream<S>&
      operator>> (istream<S>& s, bool& x)
      {
        istream_common::as_bool<bool> as_bool (x);
        return s >> as_bool;
      }


      // Floating-point
      //
      template <typename S>
      inline istream<S>&
      operator>> (istream<S>& s, float& x)
      {
        istream_common::as_float32<float> as_float32 (x);
        return s >> as_float32;
      }

      template <typename S>
      inline istream<S>&
      operator>> (istream<S>& s, double& x)
      {
        istream_common::as_float64<double> as_float64 (x);
        return s >> as_float64;
      }
    }
  }
}

#endif  // XSD_CXX_TREE_ISTREAM_HXX
