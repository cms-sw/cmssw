#ifndef HeavyFlavorAnalysis_RecoDecay_BPHGenericPtr_h
#define HeavyFlavorAnalysis_RecoDecay_BPHGenericPtr_h

#include <memory>
template <class T>
class BPHGenericPtr {
public:
  typedef typename std::shared_ptr<T> type;
  template <class... Args>
  static type create(Args&&... args) {
    return std::make_shared<T>(args...);
  }
};

#endif
