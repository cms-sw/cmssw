#ifndef Common_ClonePolicy_h
#define Common_ClonePolicy_h

namespace edm {
  template<typename T>
  struct ClonePolicy{
    static T * clone(const T & t) {
      return t.clone();
    }
  };
}

#endif
