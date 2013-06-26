#ifndef GENERS_INSERTCONTAINERITEM_HH_
#define GENERS_INSERTCONTAINERITEM_HH_

namespace gs {
    template <typename T>
    struct InsertContainerItem
    {
        static inline void insert(T& obj, const typename T::value_type& item,
                                  const std::size_t /* itemNumber */)
        {
            obj.push_back(item);
        }
    };
}

#endif // GENERS_INSERTCONTAINERITEM_HH_

