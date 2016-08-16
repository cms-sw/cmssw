#ifndef BPHGenericPtr_H
#define BPHGenericPtr_H

#include <memory>
template <class T>
class BPHGenericPtr {
 public:
  typedef typename std::shared_ptr<T> type;
};

#endif
