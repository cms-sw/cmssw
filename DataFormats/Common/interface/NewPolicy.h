#ifndef DataFormats_Common_NewPolicy_h
#define DataFormats_Common_NewPolicy_h

namespace edm {
  template<typename T>
  struct NewPolicy{
    static T * clone(const T & t) {
      return new T(t);
    }
  };
}

#endif
