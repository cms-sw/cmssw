#ifndef DataFormats_Portable_interface_PortableCollectionCommon_h
#define DataFormats_Portable_interface_PortableCollectionCommon_h

#include <cstddef>
#include <type_traits>
#include <array>

namespace portablecollection {

  // Note: if there are other uses for this, it could be moved to a central place
  template <std::size_t Start, std::size_t End, std::size_t Inc = 1, typename F>
  constexpr void constexpr_for(F&& f) {
    if constexpr (Start < End) {
      f(std::integral_constant<std::size_t, Start>());
      constexpr_for<Start + Inc, End, Inc>(std::forward<F>(f));
    }
  }

  template <std::size_t Idx, typename T>
  struct CollectionLeaf {
    CollectionLeaf() = default;
    CollectionLeaf(std::byte* buffer, int32_t elements) : layout_(buffer, elements), view_(layout_) {}
    template <std::size_t N>
    CollectionLeaf(std::byte* buffer, std::array<int32_t, N> const& sizes)
        : layout_(buffer, sizes[Idx]), view_(layout_) {
      static_assert(N >= Idx);
    }
    using Layout = T;
    using View = typename Layout::View;
    using ConstView = typename Layout::ConstView;
    Layout layout_;  //
    View view_;      //!
    // Make sure types are not void.
    static_assert(not std::is_same<T, void>::value);
  };

  template <std::size_t Idx, typename T, typename... Args>
  struct CollectionImpl : public CollectionLeaf<Idx, T>, public CollectionImpl<Idx + 1, Args...> {
    CollectionImpl() = default;
    CollectionImpl(std::byte* buffer, int32_t elements) : CollectionLeaf<Idx, T>(buffer, elements) {}

    template <std::size_t N>
    CollectionImpl(std::byte* buffer, std::array<int32_t, N> const& sizes)
        : CollectionLeaf<Idx, T>(buffer, sizes),
          CollectionImpl<Idx + 1, Args...>(CollectionLeaf<Idx, T>::layout_.metadata().nextByte(), sizes) {}
  };

  template <std::size_t Idx, typename T>
  struct CollectionImpl<Idx, T> : public CollectionLeaf<Idx, T> {
    CollectionImpl() = default;
    CollectionImpl(std::byte* buffer, int32_t elements) : CollectionLeaf<Idx, T>(buffer, elements) {}

    template <std::size_t N>
    CollectionImpl(std::byte* buffer, std::array<int32_t, N> const& sizes) : CollectionLeaf<Idx, T>(buffer, sizes) {
      static_assert(N == Idx + 1);
    }
  };

  template <typename... Args>
  struct Collections : public CollectionImpl<0, Args...> {};

  // return the type at the Idx position in Args...
  template <std::size_t Idx, typename... Args>
  using TypeResolver = typename std::tuple_element<Idx, std::tuple<Args...>>::type;

  // count how many times the type T occurs in Args...
  template <typename T, typename... Args>
  inline constexpr std::size_t typeCount = ((std::is_same<T, Args>::value ? 1 : 0) + ... + 0);

  // count the non-void elements of Args...
  template <typename... Args>
  inline constexpr std::size_t membersCount = sizeof...(Args);

  // if the type T occurs in Tuple, TupleTypeIndex has a static member value with the corresponding index;
  // otherwise there is no such data  member.
  template <typename T, typename Tuple>
  struct TupleTypeIndex {};

  template <typename T, typename... Args>
  struct TupleTypeIndex<T, std::tuple<T, Args...>> {
    static_assert(typeCount<T, Args...> == 0, "the requested type appears more than once among the arguments");
    static constexpr std::size_t value = 0;
  };

  template <typename T, typename U, typename... Args>
  struct TupleTypeIndex<T, std::tuple<U, Args...>> {
    static_assert(not std::is_same_v<T, U>);
    static_assert(typeCount<T, Args...> == 1, "the requested type does not appear among the arguments");
    static constexpr std::size_t value = 1 + TupleTypeIndex<T, std::tuple<Args...>>::value;
  };

  // if the type T occurs in Args..., TypeIndex has a static member value with the corresponding index;
  // otherwise there is no such data  member.
  template <typename T, typename... Args>
  using TypeIndex = TupleTypeIndex<T, std::tuple<Args...>>;

  // return the index where the type T occurs in Args...
  template <typename T, typename... Args>
  inline constexpr std::size_t typeIndex = TypeIndex<T, Args...>::value;

}  // namespace portablecollection

#endif  // DataFormats_Portable_interface_PortableCollectionCommon_h