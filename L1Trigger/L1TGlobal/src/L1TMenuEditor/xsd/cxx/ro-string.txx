// file      : xsd/cxx/ro-string.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

namespace xsd
{
  namespace cxx
  {
    template <typename C>
    typename ro_string<C>::size_type ro_string<C>::
    find (C c, size_type pos) const
    {
      size_type r (npos);

      if (pos < size_)
      {
        if (const C* p = traits_type::find(data_ + pos, size_ - pos, c))
          r = p - data_;
      }

      return r;
    }

    template<typename C>
    typename ro_string<C>::size_type
    trim_left (ro_string<C>& s)
    {
      typename ro_string<C>::size_type size (s.size ());

      if (size != 0)
      {
        const C* f (s.data ());
        const C* l (f + size);
        const C* of (f);

        while (f < l &&
               (*f == C (0x20) || *f == C (0x0A) ||
                *f == C (0x0D) || *f == C (0x09)))
          ++f;

        if (f != of)
        {
          size = f <= l ? l - f : 0;
          s.assign ((f <= l ? f : 0), size);
        }
      }

      return size;
    }

    template<typename C>
    typename ro_string<C>::size_type
    trim_right (ro_string<C>& s)
    {
      typename ro_string<C>::size_type size (s.size ());

      if (size != 0)
      {
        const C* f (s.data ());
        const C* l (f + size - 1);
        const C* ol (l);

        while (l > f &&
               (*l == C (0x20) || *l == C (0x0A) ||
                *l == C (0x0D) || *l == C (0x09)))
          --l;

        if (l != ol)
        {
          size = f <= l ? l - f + 1 : 0;
          s.assign ((f <= l ? f : 0), size);
        }
      }

      return size;
    }

    template<typename C>
    typename ro_string<C>::size_type
    trim (ro_string<C>& s)
    {
      typename ro_string<C>::size_type size (s.size ());

      if (size != 0)
      {
        const C* f (s.data ());
        const C* l (f + size);

        const C* of (f);

        while (f < l &&
               (*f == C (0x20) || *f == C (0x0A) ||
                *f == C (0x0D) || *f == C (0x09)))
          ++f;

        --l;

        const C* ol (l);

        while (l > f &&
               (*l == C (0x20) || *l == C (0x0A) ||
                *l == C (0x0D) || *l == C (0x09)))
          --l;

        if (f != of || l != ol)
        {
          size = f <= l ? l - f + 1 : 0;
          s.assign ((f <= l ? f : 0), size);
        }
      }

      return size;
    }

    template<typename C>
    std::basic_string<C>
    trim (const std::basic_string<C>& s)
    {
      ro_string<C> tmp (s);
      typename ro_string<C>::size_type size (tmp.size ());
      trim (tmp);

      // If we didn't change the string then return the original to help
      // avoid copying for smart (ref counted) string implementations.
      //
      if (size == tmp.size ())
        return s;
      else
        return tmp;
    }
  }
}
