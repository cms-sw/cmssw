// file      : xsd/cxx/tree/serialization/double.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_TREE_SERIALIZATION_DOUBLE_HXX
#define XSD_CXX_TREE_SERIALIZATION_DOUBLE_HXX

#include <limits> // std::numeric_limits
#include <locale>
#include <sstream>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/bits/literals.hxx>

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
        insert (const as_double<double>& d)
        {
          std::basic_string<C> r;

          if (d.x == std::numeric_limits<double>::infinity ())
            r = bits::positive_inf<C> ();
          else if (d.x == -std::numeric_limits<double>::infinity ())
            r = bits::negative_inf<C> ();
          else if (!(d.x == d.x))
            r = bits::nan<C> ();
          else
          {
            std::basic_ostringstream<C> os;
            os.imbue (std::locale::classic ());

            // Precision.
            //
#if defined (XSD_CXX_TREE_DOUBLE_PRECISION_MAX)
            os.precision (2 + std::numeric_limits<double>::digits * 301/1000);
#elif defined (XSD_CXX_TREE_DOUBLE_PRECISION)
            os.precision (XSD_CXX_TREE_DOUBLE_PRECISION);
#else
            os.precision (std::numeric_limits<double>::digits10);
#endif
            // Format.
            //
#if defined (XSD_CXX_TREE_DOUBLE_FIXED)
            os << std::fixed << d.x;
#elif defined (XSD_CXX_TREE_DOUBLE_SCIENTIFIC)
            os << std::scientific << d.x;
#else
            os << d.x;
#endif
            r = os.str ();
          }

          return r;
        }
      }

      template <typename C>
      inline void
      operator<< (list_stream<C>& ls, const as_double<double>& d)
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
              const xsd::cxx::tree::as_double<double>& d)
  {
    e << xsd::cxx::tree::bits::insert<char> (d);
  }

  inline void
  operator<< (xercesc::DOMAttr& a,
              const xsd::cxx::tree::as_double<double>& d)
  {
    a << xsd::cxx::tree::bits::insert<char> (d);
  }
}

#endif // XSD_CXX_TREE_SERIALIZATION_DOUBLE_HXX
