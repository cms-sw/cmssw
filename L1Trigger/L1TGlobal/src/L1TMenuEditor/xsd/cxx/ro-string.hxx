// file      : xsd/cxx/ro-string.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_RO_STRING_HXX
#define XSD_CXX_RO_STRING_HXX

#include <string>
#include <cstddef> // std::size_t
#include <ostream>

namespace xsd
{
  namespace cxx
  {
    // Read-only string class template.
    //
    template <typename C>
    class ro_string
    {
    public:
      typedef std::char_traits<C> traits_type;
      typedef std::size_t	    size_type;

      static const size_type npos = ~(size_type (0));

    public:
      ro_string ()
          : data_ (0), size_ (0)
      {
      }

      ro_string (const C* s)
          : data_ (s), size_ (traits_type::length (s))
      {
      }

      ro_string (const C* s, size_type size)
          : data_ (s), size_ (size)
      {
      }

      ro_string (const std::basic_string<C>& s)
          : data_ (s.data ()), size_ (s.size ())
      {
      }

      operator std::basic_string<C> () const
      {
        return std::basic_string<C> (data (), size ());
      }

    private:
      ro_string (const ro_string&);

      ro_string&
      operator= (const ro_string&);

    public:
      // The returned string is not necessarily terminated  with '\0'.
      // If size() returns 0, the returned pointer may be 0.
      //
      const C*
      data () const
      {
        return data_;
      }

      size_type
      size () const
      {
        return size_;
      }

      size_type
      length () const
      {
        return size ();
      }

    public:
      bool
      empty () const
      {
        return size () == 0;
      }

      const C&
      operator[] (size_type pos) const
      {
        return data ()[pos];
      }

    public:
      void
      assign (const C* s)
      {
        data_ = s;
        size_ = traits_type::length (s);
      }

      void
      assign (const C* s, size_type size)
      {
        data_ = s;
        size_ = size;
      }

      void
      assign (const std::basic_string<C>& s)
      {
        data_ = s.c_str ();
        size_ = s.size ();
      }

    public:
      int
      compare (const ro_string& str) const
      {
        return compare (str.data (), str.size ());
      }

      int
      compare (const std::basic_string<C>& str) const
      {
        return compare (str.c_str (), str.size ());
      }

      int
      compare (const C* str) const
      {
        return compare (str, traits_type::length (str));
      }

      int
      compare (const C* str, size_type n) const
      {
        size_type s1 (size ());
        size_type s (s1 < n ? s1 : n);

        int r (s != 0 ? traits_type::compare (data (), str, s) : 0);

        if (!r && s1 != n)
          r = s1 < n ? -1 : 1;

        return r;
      }

    public:
      size_type
      find (C c, size_type pos = 0) const;

    private:
      const C* data_;
      size_type size_;
    };

    // operator==
    //
    template <typename C>
    inline bool
    operator== (const ro_string<C>& a, const ro_string<C>& b)
    {
      return a.compare (b) == 0;
    }

    template <typename C>
    inline bool
    operator== (const ro_string<C>& a, const std::basic_string<C>& b)
    {
      return a.compare (b) == 0;
    }

    template <typename C>
    inline bool
    operator== (const std::basic_string<C>& a, const ro_string<C>& b)
    {
      return b.compare (a) == 0;
    }

    template <typename C>
    inline bool
    operator== (const ro_string<C>& a, const C* b)
    {
      return a.compare (b) == 0;
    }

    template <typename C>
    inline bool
    operator== (const C* a, const ro_string<C>& b)
    {
      return b.compare (a) == 0;
    }

    // operator!=
    //
    template <typename C>
    inline bool
    operator!= (const ro_string<C>& a, const ro_string<C>& b)
    {
      return a.compare (b) != 0;
    }

    template <typename C>
    inline bool
    operator!= (const ro_string<C>& a, const std::basic_string<C>& b)
    {
      return a.compare (b) != 0;
    }

    template <typename C>
    inline bool
    operator!= (const std::basic_string<C>& a, const ro_string<C>& b)
    {
      return b.compare (a) != 0;
    }

    template <typename C>
    inline bool
    operator!= (const ro_string<C>& a, const C* b)
    {
      return a.compare (b) != 0;
    }

    template <typename C>
    inline bool
    operator!= (const C* a, const ro_string<C>& b)
    {
      return b.compare (a) != 0;
    }

    // operator<
    //
    template <typename C>
    inline bool
    operator< (const ro_string<C>& l, const ro_string<C>& r)
    {
      return l.compare (r) < 0;
    }

    template <typename C>
    inline bool
    operator< (const ro_string<C>& l, const std::basic_string<C>& r)
    {
      return l.compare (r) < 0;
    }

    template <typename C>
    inline bool
    operator< (const std::basic_string<C>& l, const ro_string<C>& r)
    {
      return r.compare (l) > 0;
    }

    template <typename C>
    inline bool
    operator< (const ro_string<C>& l, const C* r)
    {
      return l.compare (r) < 0;
    }

    template <typename C>
    inline bool
    operator< (const C* l, const ro_string<C>& r)
    {
      return r.compare (l) > 0;
    }


    // operator>
    //
    template <typename C>
    inline bool
    operator> (const ro_string<C>& l, const ro_string<C>& r)
    {
      return l.compare (r) > 0;
    }

    template <typename C>
    inline bool
    operator> (const ro_string<C>& l, const std::basic_string<C>& r)
    {
      return l.compare (r) > 0;
    }

    template <typename C>
    inline bool
    operator> (const std::basic_string<C>& l, const ro_string<C>& r)
    {
      return r.compare (l) < 0;
    }

    template <typename C>
    inline bool
    operator> (const ro_string<C>& l, const C* r)
    {
      return l.compare (r) > 0;
    }

    template <typename C>
    inline bool
    operator> (const C* l, const ro_string<C>& r)
    {
      return r.compare (l) < 0;
    }

    // operator<=
    //
    template <typename C>
    inline bool
    operator<= (const ro_string<C>& l, const ro_string<C>& r)
    {
      return l.compare (r) <= 0;
    }

    template <typename C>
    inline bool
    operator<= (const ro_string<C>& l, const std::basic_string<C>& r)
    {
      return l.compare (r) <= 0;
    }

    template <typename C>
    inline bool
    operator<= (const std::basic_string<C>& l, const ro_string<C>& r)
    {
      return r.compare (l) >= 0;
    }

    template <typename C>
    inline bool
    operator<= (const ro_string<C>& l, const C* r)
    {
      return l.compare (r) <= 0;
    }

    template <typename C>
    inline bool
    operator<= (const C* l, const ro_string<C>& r)
    {
      return r.compare (l) >= 0;
    }


    // operator>=
    //
    template <typename C>
    inline bool
    operator>= (const ro_string<C>& l, const ro_string<C>& r)
    {
      return l.compare (r) >= 0;
    }

    template <typename C>
    inline bool
    operator>= (const ro_string<C>& l, const std::basic_string<C>& r)
    {
      return l.compare (r) >= 0;
    }

    template <typename C>
    inline bool
    operator>= (const std::basic_string<C>& l, const ro_string<C>& r)
    {
      return r.compare (l) <= 0;
    }

    template <typename C>
    inline bool
    operator>= (const ro_string<C>& l, const C* r)
    {
      return l.compare (r) >= 0;
    }

    template <typename C>
    inline bool
    operator>= (const C* l, const ro_string<C>& r)
    {
      return r.compare (l) <= 0;
    }

    // operator<<
    //
    template<typename C>
    std::basic_ostream<C>&
    operator<< (std::basic_ostream<C>& os, const ro_string<C>& str)
    {
      if (str.size () != 0)
        os.write (str.data (), static_cast<std::streamsize> (str.size ()));

      return os;
    }

    // operator+=
    //
    template<typename C>
    std::basic_string<C>&
    operator+= (std::basic_string<C>& l, const ro_string<C>& r)
    {
      l.append (r.data (), r.size ());
      return l;
    }

    // Trim leading and trailing XML whitespaces. Return the new
    // string size.
    //
    template <typename C>
    typename ro_string<C>::size_type
    trim_left (ro_string<C>&);

    template <typename C>
    typename ro_string<C>::size_type
    trim_right (ro_string<C>&);

    template <typename C>
    typename ro_string<C>::size_type
    trim (ro_string<C>&);

    // Trim leading and trailing XML whitespaces.
    //
    template<typename C>
    std::basic_string<C>
    trim (const std::basic_string<C>&);
  }
}

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/ro-string.txx>

#endif  // XSD_CXX_RO_STRING_HXX
