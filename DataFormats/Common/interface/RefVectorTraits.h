#ifndef DataFormats_Common_RefVectorTrait_h
#define DataFormats_Common_RefVectorTrait_h

namespace edm {
  template<typename C, typename T, typename F> class Ref;
  template<typename C, typename T, typename F> class RefVector;
  template<typename C, typename T, typename F> class RefVectorIterator;
  namespace refhelper {
    template<typename C, typename T, typename F>
    struct RefVectorTrait {
      typedef Ref<C, T, F> ref_type;
      typedef RefVectorIterator<C, T, F> iterator_type;
    };

    template<typename C, typename T, typename F, typename T1, typename F1>
     struct RefVectorTrait<RefVector<C, T, F>, T1, F1> {
      typedef Ref<C, T, F> ref_type;
      typedef RefVectorIterator<C, T, F> iterator_type;
    };

  }
}

#endif
