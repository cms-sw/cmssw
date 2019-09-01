#ifndef DataFormats_Common_RefTraits_h
#define DataFormats_Common_RefTraits_h

#include <algorithm>

namespace edm {
  template <typename C, typename T, typename F>
  class RefVector;
  template <typename T>
  class RefToBaseVector;

  namespace refhelper {
    template <typename C, typename T>
    struct FindUsingAdvance {
      typedef C const& first_argument_type;
      typedef unsigned int second_argument_type;
      typedef T const* result_type;

      result_type operator()(first_argument_type iContainer, second_argument_type iIndex) {
        typename C::const_iterator it = iContainer.begin();
        std::advance(it, static_cast<typename C::size_type>(iIndex));
        return it.operator->();
      }
    };

    template <typename REFV>
    struct FindRefVectorUsingAdvance {
      using first_argument_type = REFV const&;
      using second_argument_type = typename REFV::key_type;
      using result_type = typename REFV::member_type const*;

      result_type operator()(first_argument_type iContainer, second_argument_type iIndex) {
        typename REFV::const_iterator it = iContainer.begin();
        std::advance(it, iIndex);
        return it.operator->()->get();
      }
    };

    //Used in edm::Ref to set the default 'find' method to use based on the Container and 'contained' type
    template <typename C, typename T>
    struct FindTrait {
      typedef FindUsingAdvance<C, T> value;
    };

    template <typename C, typename T, typename F>
    struct FindTrait<RefVector<C, T, F>, T> {
      typedef FindRefVectorUsingAdvance<RefVector<C, T, F> > value;
    };

    template <typename T>
    struct FindTrait<RefToBaseVector<T>, T> {
      typedef FindRefVectorUsingAdvance<RefToBaseVector<T> > value;
    };

    template <typename C>
    struct ValueTrait {
      typedef typename C::value_type value;
    };

    template <typename C, typename T, typename F>
    struct ValueTrait<RefVector<C, T, F> > {
      typedef T value;
    };

    template <typename T>
    struct ValueTrait<RefToBaseVector<T> > {
      typedef T value;
    };

  }  // namespace refhelper

}  // namespace edm
#endif
