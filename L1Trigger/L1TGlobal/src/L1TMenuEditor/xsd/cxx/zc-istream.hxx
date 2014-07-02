// file      : xsd/cxx/zc-istream.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_ZC_ISTREAM_HXX
#define XSD_CXX_ZC_ISTREAM_HXX

#include <string>
#include <istream>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/ro-string.hxx>

namespace xsd
{
  namespace cxx
  {
    // Input streambuffer that does not copy the underlying
    // buffer (zero copy).
    //
    template <typename C>
    class zc_streambuf: public std::basic_streambuf<C>
    {
    public:
      typedef typename std::basic_streambuf<C>::int_type int_type;
      typedef typename std::basic_streambuf<C>::traits_type traits_type;

    public:
      zc_streambuf (const ro_string<C>&);
      zc_streambuf (const std::basic_string<C>&);

    protected:
      virtual std::streamsize
      showmanyc ();

      virtual int_type
      underflow ();

    private:
      void
      init ();

    private:
      zc_streambuf (const zc_streambuf&);

      zc_streambuf&
      operator= (const zc_streambuf&);

    private:
      ro_string<C> str_;
    };


    // Input string stream that does not copy the underlying string.
    //
    template <typename C>
    class zc_istream_base
    {
    protected:
      zc_istream_base (const ro_string<C>&);
      zc_istream_base (const std::basic_string<C>&);

    protected:
      zc_streambuf<C> buf_;
    };

    template <typename C>
    class zc_istream: protected zc_istream_base<C>,
                      public std::basic_istream<C>
    {
      typedef std::basic_istream<C> base;

    public:
      zc_istream (const ro_string<C>&);
      zc_istream (const std::basic_string<C>&);

      bool
      exhausted ()
      {
        return this->get () == std::basic_istream<C>::traits_type::eof ();
      }

      zc_istream&
      operator>> (unsigned char& x)
      {
        if (check_unsigned ())
          static_cast<base&> (*this) >> x;

        return *this;
      }

      zc_istream&
      operator>> (signed char& x)
      {
        if (check_signed ())
          static_cast<base&> (*this) >> x;

        return *this;
      }

      zc_istream&
      operator>> (unsigned short& x)
      {
        if (check_unsigned ())
          static_cast<base&> (*this) >> x;

        return *this;
      }

      zc_istream&
      operator>> (short& x)
      {
        if (check_signed ())
          static_cast<base&> (*this) >> x;

        return *this;
      }

      zc_istream&
      operator>> (unsigned int& x)
      {
        if (check_unsigned ())
          static_cast<base&> (*this) >> x;

        return *this;
      }

      zc_istream&
      operator>> (int& x)
      {
        if (check_signed ())
          static_cast<base&> (*this) >> x;

        return *this;
      }

      zc_istream&
      operator>> (unsigned long& x)
      {
        if (check_unsigned ())
          static_cast<base&> (*this) >> x;

        return *this;
      }

      zc_istream&
      operator>> (long& x)
      {
        if (check_signed ())
          static_cast<base&> (*this) >> x;

        return *this;
      }

      zc_istream&
      operator>> (unsigned long long& x)
      {
        if (check_unsigned ())
          static_cast<base&> (*this) >> x;

        return *this;
      }

      zc_istream&
      operator>> (long long& x)
      {
        if (check_signed ())
          static_cast<base&> (*this) >> x;

        return *this;
      }

      template <typename X>
      zc_istream&
      operator>> (X& x)
      {
        static_cast<base&> (*this) >> x;
        return *this;
      }

    private:
      bool
      check_signed ()
      {
        typename std::basic_istream<C>::traits_type::int_type p (this->peek ());
        bool r ((p >= C ('0') && p <= C ('9')) || p == C ('-') || p == C ('+'));

        if (!r)
          this->setstate (std::ios_base::failbit);

        return r;
      }

      bool
      check_unsigned ()
      {
        typename std::basic_istream<C>::traits_type::int_type p (this->peek ());
        bool r ((p >= C ('0') && p <= C ('9')) || p == C ('+'));

        if (!r)
          this->setstate (std::ios_base::failbit);

        return r;
      }

    private:
      zc_istream (const zc_istream&);

      zc_istream&
      operator= (const zc_istream&);
    };
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/zc-istream.txx>

#endif  // XSD_CXX_ZC_ISTREAM_HXX
