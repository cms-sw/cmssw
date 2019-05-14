#ifndef GENERS_INSERTCONTAINERITEM_HH_
#define GENERS_INSERTCONTAINERITEM_HH_

#include <cstddef>

namespace gs {
  template <typename T>
  struct InsertContainerItem {
    static inline void insert(T &obj, const typename T::value_type &item, const std::size_t /* itemNumber */) {
      obj.push_back(item);
    }
  };
}  // namespace gs

#endif  // GENERS_INSERTCONTAINERITEM_HH_
