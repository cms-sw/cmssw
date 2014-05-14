// file      : xsd/cxx/tree/parsing/element-map.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_TREE_PARSING_ELEMENT_MAP_TXX
#define XSD_CXX_TREE_PARSING_ELEMENT_MAP_TXX

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/xml/dom/elements.hxx>

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/exceptions.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      template <typename C, typename T>
      std::auto_ptr<element_type<C, T> > element_map<C, T>::
      parse (const xercesc::DOMElement& e, flags f)
      {
        const qualified_name n (xml::dom::name<C> (e));
        typename map::const_iterator i (map_->find (n));

        if (i != map_->end () && i->second.parser_ != 0)
          return (i->second.parser_) (e, f);
        else
          throw no_element_info<C> (n.name (), n.namespace_ ());
      }

      template<typename T, typename C, typename B>
      std::auto_ptr<element_type<C, B> >
      parser_impl (const xercesc::DOMElement& e, flags f)
      {
        return std::auto_ptr<element_type<C, B> > (new T (e, f));
      }
    }
  }
}

#endif // XSD_CXX_TREE_PARSING_ELEMENT_MAP_TXX
