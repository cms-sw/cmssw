// file      : xsd/cxx/tree/serialization/element-map.txx
// author    : Boris Kolpackov <boris@codesynthesis.com>
// copyright : Copyright (c) 2005-2010 Code Synthesis Tools CC
// license   : GNU GPL v2 + exceptions; see accompanying LICENSE file

#ifndef XSD_CXX_TREE_SERIALIZATION_ELEMENT_MAP_TXX
#define XSD_CXX_TREE_SERIALIZATION_ELEMENT_MAP_TXX

#include <L1Trigger/L1TGlobal/src/L1TMenuEditor/xsd/cxx/tree/exceptions.hxx>

namespace xsd
{
  namespace cxx
  {
    namespace tree
    {
      template <typename C, typename T>
      void element_map<C, T>::
      serialize (xercesc::DOMElement& e, const element_type& x)
      {
        const qualified_name n (x._name (), x._namespace ());
        typename map::const_iterator i (map_->find (n));

        if (i != map_->end () && i->second.serializer_ != 0)
          return (i->second.serializer_) (e, x);
        else
          throw no_element_info<C> (n.name (), n.namespace_ ());
      }

      template<typename T, typename C, typename B>
      void
      serializer_impl (xercesc::DOMElement& e, const element_type<C, B>& x)
      {
        e << static_cast<const T&> (x);
      }
    }
  }
}

#endif // XSD_CXX_TREE_SERIALIZATION_ELEMENT_MAP_TXX
