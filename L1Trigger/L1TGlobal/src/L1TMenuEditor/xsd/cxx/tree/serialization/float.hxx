// file      : xsd/cxx/tree/serialization/float.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_TREE_SERIALIZATION_FLOAT_HXX
#define XSD_CXX_TREE_SERIALIZATION_FLOAT_HXX

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
        insert (float f)
        {
          std::basic_string<C> r;

          if (f == std::numeric_limits<float>::infinity ())
            r = bits::positive_inf<C> ();
          else if (f == -std::numeric_limits<float>::infinity ())
            r = bits::negative_inf<C> ();
          else if (!(f == f))
            r = bits::nan<C> ();
          else
          {
            std::basic_ostringstream<C> os;
            os.imbue (std::locale::classic ());

            // Precision.
            //
#if defined (XSD_CXX_TREE_FLOAT_PRECISION_MAX)
            os.precision (2 + std::numeric_limits<float>::digits * 301/1000);
#elif defined (XSD_CXX_TREE_FLOAT_PRECISION)
            os.precision (XSD_CXX_TREE_FLOAT_PRECISION);
#else
            os.precision (std::numeric_limits<float>::digits10);
#endif
            // Format.
            //
#if defined (XSD_CXX_TREE_FLOAT_FIXED)
            os << std::fixed << f;
#elif defined (XSD_CXX_TREE_FLOAT_SCIENTIFIC)
            os << std::scientific << f;
#else
            os << f;
#endif
            r = os.str ();
          }

          return r;
        }
      }

      template <typename C>
      inline void
      operator<< (list_stream<C>& ls, float f)
      {
        ls.os_ << bits::insert<C> (f);
      }
    }
  }
}

namespace XERCES_CPP_NAMESPACE
{
  inline void
  operator<< (xercesc::DOMElement& e, float f)
  {
    e << xsd::cxx::tree::bits::insert<char> (f);
  }

  inline void
  operator<< (xercesc::DOMAttr& a, float f)
  {
    a << xsd::cxx::tree::bits::insert<char> (f);
  }
}

#endif // XSD_CXX_TREE_SERIALIZATION_FLOAT_HXX
