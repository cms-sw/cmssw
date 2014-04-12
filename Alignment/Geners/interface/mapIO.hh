#ifndef GENERS_MAPIO_HH_
#define GENERS_MAPIO_HH_

#include <map>
#include "Alignment/Geners/interface/GenericIO.hh"

namespace gs {
    template <class Key, class T, class Compare, class Alloc>
    struct InsertContainerItem<std::map<Key,T,Compare,Alloc> >
    {
        typedef std::map<Key,T,Compare,Alloc> A;
        static inline void insert(A& obj, const typename A::value_type& item,
                                  const std::size_t /* itemNumber */)
        {obj.insert(item);}
    };

    template <class Key, class T, class Compare, class Alloc>
    struct InsertContainerItem<volatile std::map<Key,T,Compare,Alloc> >
    {
        typedef std::map<Key,T,Compare,Alloc> A;
        static inline void insert(A& obj, const typename A::value_type& item,
                                  const std::size_t /* itemNumber */)
        {obj.insert(item);}
    };

    template <class Key, class T, class Compare, class Alloc>
    struct InsertContainerItem<std::multimap<Key,T,Compare,Alloc> >
    {
        typedef std::multimap<Key,T,Compare,Alloc> A;
        static inline void insert(A& obj, const typename A::value_type& item,
                                  const std::size_t /* itemNumber */)
        {obj.insert(item);}
    };

    template <class Key, class T, class Compare, class Alloc>
    struct InsertContainerItem<volatile std::multimap<Key,T,Compare,Alloc> >
    {
        typedef std::multimap<Key,T,Compare,Alloc> A;
        static inline void insert(A& obj, const typename A::value_type& item,
                                  const std::size_t /* itemNumber */)
        {obj.insert(item);}
    };
}

gs_specialize_template_id_TTTT(std::map, 0, 3)
gs_specialize_template_id_TTTT(std::multimap, 0, 3)

#endif // GENERS_MAPIO_HH_

