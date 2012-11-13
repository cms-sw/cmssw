#ifndef GENERS_SETIO_HH_
#define GENERS_SETIO_HH_

#include <set>
#include "Alignment/Geners/interface/GenericIO.hh"

namespace gs {
    template <class T, class Compare, class Alloc>
    struct InsertContainerItem<std::set<T,Compare,Alloc> >
    {
        typedef std::set<T,Compare,Alloc> A;
        static inline void insert(A& obj, const typename A::value_type& item,
                                  const std::size_t /* itemNumber */)
        {obj.insert(item);}
    };

    template <class T, class Compare, class Alloc>
    struct InsertContainerItem<volatile std::set<T,Compare,Alloc> >
    {
        typedef std::set<T,Compare,Alloc> A;
        static inline void insert(A& obj, const typename A::value_type& item,
                                  const std::size_t /* itemNumber */)
        {obj.insert(item);}
    };

    template <class T, class Compare, class Alloc>
    struct InsertContainerItem<std::multiset<T,Compare,Alloc> >
    {
        typedef std::multiset<T,Compare,Alloc> A;
        static inline void insert(A& obj, const typename A::value_type& item,
                                  const std::size_t /* itemNumber */)
        {obj.insert(item);}
    };

    template <class T, class Compare, class Alloc>
    struct InsertContainerItem<volatile std::multiset<T,Compare,Alloc> >
    {
        typedef std::multiset<T,Compare,Alloc> A;
        static inline void insert(A& obj, const typename A::value_type& item,
                                  const std::size_t /* itemNumber */)
        {obj.insert(item);}
    };
}

gs_specialize_template_id_TTT(std::set, 0, 2)
gs_specialize_template_id_TTT(std::multiset, 0, 2)

#endif // GENERS_SETIO_HH_

