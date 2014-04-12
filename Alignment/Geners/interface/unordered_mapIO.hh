#ifndef GENERS_UNORDERED_MAPIO_HH_
#define GENERS_UNORDERED_MAPIO_HH_

#include "Alignment/Geners/interface/CPP11_config.hh"
#ifdef CPP11_STD_AVAILABLE

#include <unordered_map>
#include "Alignment/Geners/interface/GenericIO.hh"
#include "Alignment/Geners/interface/specialize_hash_io.hh"

namespace gs {
    template <class Key, class T, class Hash, class Pred, class Alloc>
    struct InsertContainerItem<std::unordered_map<Key,T,Hash,Pred,Alloc> >
    {
        typedef std::unordered_map<Key,T,Hash,Pred,Alloc> A;
        static inline void insert(A& obj, const typename A::value_type& item,
                                  const std::size_t /* itemNumber */)
        {obj.insert(item);}
    };

    template <class Key, class T, class Hash, class Pred, class Alloc>
    struct InsertContainerItem<volatile std::unordered_map<Key,T,Hash,Pred,Alloc> >
    {
        typedef std::unordered_map<Key,T,Hash,Pred,Alloc> A;
        static inline void insert(A& obj, const typename A::value_type& item,
                                  const std::size_t /* itemNumber */)
        {obj.insert(item);}
    };

    template <class Key, class T, class Hash, class Pred, class Alloc>
    struct InsertContainerItem<std::unordered_multimap<Key,T,Hash,Pred,Alloc> >
    {
        typedef std::unordered_multimap<Key,T,Hash,Pred,Alloc> A;
        static inline void insert(A& obj, const typename A::value_type& item,
                                  const std::size_t /* itemNumber */)
        {obj.insert(item);}
    };

    template <class Key, class T, class Hash, class Pred, class Alloc>
    struct InsertContainerItem<volatile std::unordered_multimap<Key,T,Hash,Pred,Alloc> >
    {
        typedef std::unordered_multimap<Key,T,Hash,Pred,Alloc> A;
        static inline void insert(A& obj, const typename A::value_type& item,
                                  const std::size_t /* itemNumber */)
        {obj.insert(item);}
    };
}

gs_specialize_template_id_TTTTT(std::unordered_map, 0, 4)
gs_specialize_template_id_TTTTT(std::unordered_multimap, 0, 4)

#endif // CPP11_STD_AVAILABLE
#endif // GENERS_UNORDERED_MAPIO_HH_

