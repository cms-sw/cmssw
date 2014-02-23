// file      : xsd/cxx/tree/serialization/decimal.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_TREE_SERIALIZATION_DECIMAL_HXX
#define XSD_CXX_TREE_SERIALIZATION_DECIMAL_HXX

#include <limits> // std::numeric_limits
#include <locale>
#include <sstream>

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      namespace bits
      {
        // The formula for the number of decimla digits required is given in:
        //
        // http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2005/n1822.pdf
        //
        template <typename C>
        std::basic_string<C>
        insert (const as_decimal<double>& d)
        {
          std::basic_ostringstream<C> os;
          os.imbue (std::locale::classic ());
          std::streamsize prec;

          const facet* f = d.facets ?
            facet::find (d.facets, facet::fraction_digits) : 0;

          if (f)
            prec = static_cast<std::streamsize> (f->value);
          else
          {
            // Precision.
            //
#if defined (XSD_CXX_TREE_DECIMAL_PRECISION_MAX)
            prec = 2 + std::numeric_limits<double>::digits * 301/1000;
#elif defined (XSD_CXX_TREE_DECIMAL_PRECISION)
            prec = XSD_CXX_TREE_DECIMAL_PRECISION;
#else
            prec = std::numeric_limits<double>::digits10;
#endif
          }

          os.precision (prec);
          os << std::fixed << d.x;
          std::basic_string<C> r (os.str ());
          const C* cr (r.c_str ());

          // Remove the trailing zeros and the decimal point if necessary.
          //
          typename std::basic_string<C>::size_type size (r.size ()), n (size);

          if (prec != 0)
          {
            for (; n > 0 && cr[n - 1] == '0'; --n)/*noop*/;

            if (n > 0 && cr[n - 1] == '.')
              --n;
          }

          // See if we have a restriction on total digits.
          //
          f = d.facets ? facet::find (d.facets, facet::total_digits) : 0;

          if (f && n > f->value)
          {
            // Point and sign do not count so figure out if we have them.
            //
            typename std::basic_string<C>::size_type extra (
              cr[0] == '-' ? 1 : 0);

            if (r.find ('.') < n)
              extra++;

            // Unless we have a point and the size difference is one,
            // remove some digits.
            //
            if ((n - extra) > f->value)
              n -= (n - extra - f->value);

            if (n > 0 && cr[n - 1] == '.')
              --n;
          }

          if (n != size)
            r.resize (n);

          return r;
        }
      }

      template <typename C>
      inline void
      operator<< (list_stream<C>& ls, const as_decimal<double>& d)
      {
        ls.os_ << bits::insert<C> (d);
      }
    }
  }
}

namespace XERCES_CPP_NAMESPACE
{
  inline void
  operator<< (xercesc::DOMElement& e,
              const xsd::cxx::tree::as_decimal<double>& d)
  {
    e << xsd::cxx::tree::bits::insert<char> (d);
  }

  inline void
  operator<< (xercesc::DOMAttr& a,
              const xsd::cxx::tree::as_decimal<double>& d)
  {
    a << xsd::cxx::tree::bits::insert<char> (d);
  }
}

#endif // XSD_CXX_TREE_SERIALIZATION_DECIMAL_HXX
