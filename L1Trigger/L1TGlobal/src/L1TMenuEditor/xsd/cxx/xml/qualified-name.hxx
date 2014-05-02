// file      : xsd/cxx/xml/qualified-name.hxx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_XML_QUALIFIED_NAME_HXX
#define XSD_CXX_XML_QUALIFIED_NAME_HXX

#include <string>

namespace xsd
{
  namespace cxx
  {
    namespace xml
    {
      template <typename C>
      struct qualified_name
      {
        qualified_name (const C* name,
                        const C* namespace_)
            : name_ (name), namespace__ (namespace_)
        {
        }

        qualified_name (const std::basic_string<C>& name,
                        const std::basic_string<C>& namespace_)
            : name_ (name), namespace__ (namespace_)
        {
        }

        qualified_name (const C* name)
            : name_ (name)
        {
        }

        qualified_name (const std::basic_string<C>& name)
            : name_ (name)
        {
        }

        const std::basic_string<C>&
        name () const
        {
          return name_;
        }

        const std::basic_string<C>&
        namespace_ () const
        {
          return namespace__;
        }

      private:
        std::basic_string<C> name_;
        std::basic_string<C> namespace__;
      };

      template <typename C>
      inline bool
      operator== (const qualified_name<C>& x, const qualified_name<C>& y)
      {
        return x.name () == y.name () && x.namespace_ () == y.namespace_ ();
      }

      template <typename C>
      inline bool
      operator!= (const qualified_name<C>& x, const qualified_name<C>& y)
      {
        return !(x == y);
      }

      template <typename C>
      inline bool
      operator< (const qualified_name<C>& x, const qualified_name<C>& y)
      {
        int r (x.name ().compare (y.name ()));
        return (r < 0) || (r == 0 && x.namespace_ () < y.namespace_ ());
      }
    }
  }
}

#endif // XSD_CXX_XML_QUALIFIED_NAME_HXX
