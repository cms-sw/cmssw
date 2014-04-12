#ifndef GENERS_UNORDERED_SETIO_HH_
#define GENERS_UNORDERED_SETIO_HH_

#include "Alignment/Geners/interface/CPP11_config.hh"
#ifdef CPP11_STD_AVAILABLE

#include <unordered_set>
#include "Alignment/Geners/interface/GenericIO.hh"
#include "Alignment/Geners/interface/specialize_hash_io.hh"

namespace gs {
    template <class T, class Hash, class Pred, class Alloc>
    struct InsertContainerItem<std::unordered_set<T,Hash,Pred,Alloc> >
    {
        typedef std::unordered_set<T,Hash,Pred,Alloc> A;
        static inline void insert(A& obj, const typename A::value_type& item,
                                  const std::size_t /* itemNumber */)
        {obj.insert(item);}
    };

    template <class T, class Hash, class Pred, class Alloc>
    struct InsertContainerItem<volatile std::unordered_set<T,Hash,Pred,Alloc> >
    {
        typedef std::unordered_set<T,Hash,Pred,Alloc> A;
        static inline void insert(A& obj, const typename A::value_type& item,
                                  const std::size_t /* itemNumber */)
        {obj.insert(item);}
    };

    template <class T, class Hash, class Pred, class Alloc>
    struct InsertContainerItem<std::unordered_multiset<T,Hash,Pred,Alloc> >
    {
        typedef std::unordered_multiset<T,Hash,Pred,Alloc> A;
        static inline void insert(A& obj, const typename A::value_type& item,
                                  const std::size_t /* itemNumber */)
        {obj.insert(item);}
    };

    template <class T, class Hash, class Pred, class Alloc>
    struct InsertContainerItem<volatile std::unordered_multiset<T,Hash,Pred,Alloc> >
    {
        typedef std::unordered_multiset<T,Hash,Pred,Alloc> A;
        static inline void insert(A& obj, const typename A::value_type& item,
                                  const std::size_t /* itemNumber */)
        {obj.insert(item);}
    };
}

gs_specialize_template_id_TTTT(std::unordered_set, 0, 2)
gs_specialize_template_id_TTTT(std::unordered_multiset, 0, 2)

#endif // CPP11_STD_AVAILABLE
#endif // GENERS_UNORDERED_SETIO_HH_

