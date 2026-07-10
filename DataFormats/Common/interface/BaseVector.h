#ifndef DataFormats_Common_BaseVector_h
#define DataFormats_Common_BaseVector_h

/**
 * \class BaseVector
 * description:
 *  A vector of objects of a common base class, but different derived types.  
 *  The derived types must be specified in the Traits class passed as the second template argument.
 *  E.g.
 * \code
 *   struct Base {};
 *   struct DerivedA : public Base {};
 *   struct DerivedB : public Base {};
 *   template <> struct BaseVectorTraits<Base> {
 *     using variant_type = std::variant<DerivedA, DerivedB>;
 *   };
 * \endcode
 */

#include <variant>
#include <vector>

namespace edm {

  template <typename T>
  struct BaseVectorTraits;

  namespace detail {
    template <typename R, typename U>
    constexpr R& getBase(U&& data) noexcept {
      auto v = [](auto&& value) -> R& { return static_cast<R&>(value); };
      return std::visit(v, data);
    }
  }  // namespace detail
  template <typename T, typename Traits = BaseVectorTraits<T>>
    requires requires { typename Traits::variant_type; }
  class BaseVector {
  public:
    using value_type = T;
    using reference_type = value_type&;
    using variant_type = typename Traits::variant_type;

    constexpr BaseVector() noexcept = default;
    constexpr BaseVector(BaseVector const&) = default;
    constexpr BaseVector(BaseVector&&) noexcept = default;
    constexpr BaseVector& operator=(BaseVector const&) = default;
    constexpr BaseVector& operator=(BaseVector&&) noexcept = default;

    constexpr auto size() const noexcept { return data_.size(); }
    constexpr auto max_size() const noexcept { return data_.max_size(); }
    constexpr auto empty() const noexcept { return data_.empty(); }
    constexpr auto capacity() const noexcept { return data_.capacity(); }
    constexpr void reserve(size_t new_cap) noexcept { data_.reserve(new_cap); }
    constexpr void shrink_to_fit() noexcept { data_.shrink_to_fit(); }
    constexpr void clear() noexcept { data_.clear(); }

    template <typename U>
    constexpr void push_back(U&& value) {
      data_.emplace_back(std::forward<U>(value));
    }
    template <typename U, typename... A>
    constexpr void emplace_back(A&&... value) {
      data_.emplace_back(U{std::forward<A>(value)...});
    }

    constexpr void pop_back() noexcept { data_.pop_back(); }
    constexpr void swap(BaseVector& other) noexcept { data_.swap(other.data_); }

    constexpr value_type const& operator[](size_t i) const noexcept {
      return detail::getBase<value_type const>(data_[i]);
    }
    constexpr value_type& operator[](size_t i) noexcept { return detail::getBase<value_type>(data_[i]); }

    constexpr value_type const& front() const noexcept { return detail::getBase<value_type const>(data_.front()); }
    constexpr value_type& front() noexcept { return detail::getBase<value_type>(data_.front()); }
    constexpr value_type const& back() const noexcept { return detail::getBase<value_type const>(data_.back()); }
    constexpr value_type& back() noexcept { return detail::getBase<value_type>(data_.back()); }

    /// @brief If the object at index i is of type U, return a pointer to it, otherwise return nullptr
    /// @tparam U type of the object to get. Must be one of the types in the variant_type of the BaseVectorTraits specialization for T.
    /// @param i index of the object to get
    /// @return Pointer to the object if it is of type U, otherwise nullptr
    template <typename U>
    constexpr auto get_if(size_t i) const noexcept {
      return std::get_if<U>(&data_[i]);
    }

    class const_iterator;
    class iterator {
    public:
      friend class BaseVector<T, Traits>;
      friend class const_iterator;
      using iterator_category = std::random_access_iterator_tag;
      using value_type = T;
      using difference_type = std::ptrdiff_t;
      using pointer = value_type*;
      using reference = value_type&;
      constexpr iterator() noexcept = default;
      constexpr iterator(iterator const&) = default;
      constexpr iterator& operator=(iterator const&) = default;
      constexpr iterator(iterator&&) noexcept = default;
      constexpr iterator& operator=(iterator&&) noexcept = default;

      constexpr explicit iterator(std::vector<variant_type>::iterator it) noexcept : it_(it) {}
      constexpr reference operator*() noexcept { return detail::getBase<value_type>((*it_)); }
      constexpr pointer operator->() noexcept { return &detail::getBase<value_type>((*it_)); }
      constexpr iterator& operator++() noexcept {
        ++it_;
        return *this;
      }
      constexpr iterator operator++(int) noexcept {
        iterator tmp = *this;
        ++it_;
        return tmp;
      }
      constexpr iterator& operator--() noexcept {
        --it_;
        return *this;
      }
      constexpr iterator operator--(int) noexcept {
        iterator tmp = *this;
        --it_;
        return tmp;
      }
      constexpr iterator operator+(difference_type n) const noexcept { return iterator(it_ + n); }
      constexpr iterator operator-(difference_type n) const noexcept { return iterator(it_ - n); }
      constexpr difference_type operator-(iterator const& other) const noexcept { return it_ - other.it_; }

      constexpr auto operator<=>(iterator const& other) const noexcept = default;

    private:
      std::vector<variant_type>::iterator it_;
    };
    class const_iterator {
    public:
      friend class BaseVector<T, Traits>;
      using iterator_category = std::random_access_iterator_tag;
      using value_type = T const;
      using difference_type = std::ptrdiff_t;
      using pointer = value_type*;
      using reference = value_type&;
      constexpr const_iterator() noexcept = default;
      constexpr const_iterator(const_iterator const&) = default;
      constexpr const_iterator& operator=(const_iterator const&) = default;
      constexpr const_iterator(const_iterator&&) noexcept = default;
      constexpr const_iterator& operator=(const_iterator&&) noexcept = default;

      constexpr const_iterator(iterator const& it) noexcept : it_(it.it_) {}

      constexpr explicit const_iterator(std::vector<variant_type>::const_iterator it) noexcept : it_(it) {}
      constexpr reference operator*() const noexcept { return detail::getBase<value_type>((*it_)); }
      constexpr pointer operator->() const noexcept { return &detail::getBase<value_type>((*it_)); }
      constexpr const_iterator& operator++() noexcept {
        ++it_;
        return *this;
      }
      constexpr const_iterator operator++(int) noexcept {
        const_iterator tmp = *this;
        ++it_;
        return tmp;
      }
      constexpr const_iterator& operator--() noexcept {
        --it_;
        return *this;
      }
      constexpr const_iterator operator--(int) noexcept {
        const_iterator tmp = *this;
        --it_;
        return tmp;
      }
      constexpr const_iterator operator+(difference_type n) const noexcept { return const_iterator(it_ + n); }
      constexpr const_iterator operator-(difference_type n) const noexcept { return const_iterator(it_ - n); }
      constexpr difference_type operator-(const_iterator const& other) const noexcept { return it_ - other.it_; }

      constexpr auto operator<=>(const_iterator const& other) const noexcept = default;

    private:
      std::vector<variant_type>::const_iterator it_;
    };

    template <typename U>
    constexpr iterator insert(const_iterator pos, U&& value) {
      return iterator(data_.emplace(pos.it_, std::forward<U>(value)));
    }
    template <typename U, typename... A>
    constexpr iterator emplace(const_iterator pos, A&&... value) {
      return iterator(data_.emplace(pos.it_, U{std::forward<A>(value)...}));
    }

    constexpr void erase(const_iterator pos) noexcept { data_.erase(pos.it_); }
    constexpr void erase(const_iterator first, const_iterator last) noexcept { data_.erase(first.it_, last.it_); }

    constexpr iterator begin() noexcept { return iterator(data_.begin()); }
    constexpr iterator end() noexcept { return iterator(data_.end()); }
    constexpr const_iterator begin() const noexcept { return const_iterator(data_.begin()); }
    constexpr const_iterator end() const noexcept { return const_iterator(data_.end()); }

    constexpr const_iterator cbegin() const noexcept { return const_iterator(data_.cbegin()); }
    constexpr const_iterator cend() const noexcept { return const_iterator(data_.cend()); }

    // Expose the underlying variant iterators for use in algorithms that need to access the variant directly
    constexpr auto variant_begin() noexcept { return data_.begin(); }
    constexpr auto variant_end() noexcept { return data_.end(); }
    constexpr auto variant_begin() const noexcept { return data_.cbegin(); }
    constexpr auto variant_end() const noexcept { return data_.cend(); }

  private:
    std::vector<variant_type> data_;
  };

}  // namespace edm

#endif
