#ifndef RecoTracker_MkFitCore_interface_radix_sort_h
#define RecoTracker_MkFitCore_interface_radix_sort_h

#include <type_traits>
#include <vector>

namespace mkfit {

  template <typename V, typename R>
  class radix_sort {
  public:
    static_assert(std::is_same<V, unsigned short>() || std::is_same<V, unsigned int>());
    static_assert(std::is_same<R, unsigned short>() || std::is_same<R, unsigned int>());

    typedef unsigned char ubyte_t;
    typedef V value_t;
    typedef R rank_t;
    static constexpr rank_t c_NBytes = sizeof(V);

    void sort(const std::vector<V>& values, std::vector<R>& ranks);

  private:
    void histo_loop(const std::vector<V>& values, rank_t* histos);
    void radix_loop(const std::vector<V>& values, rank_t* histos, std::vector<R>& ranks);
  };
}  // namespace mkfit

#endif
