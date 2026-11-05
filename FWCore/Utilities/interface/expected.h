#ifndef FWCore_Utilities_expected_h
#define FWCore_Utilities_expected_h

#include <cassert>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace edm {

  // A simple wrapper to distinguish the error type from the value type
  template <typename E>
  class unexpected {
  public:
    template <typename Err = E>
      requires(not std::is_same_v<std::remove_cvref_t<Err>, unexpected> and
               not std::is_same_v<std::remove_cvref_t<Err>, std::in_place_t> and std::is_constructible_v<E, Err>)
    constexpr explicit unexpected(Err&& e) : error_(std::forward<Err>(e)) {}

    constexpr const E& error() const& noexcept { return error_; }
    constexpr E& error() & noexcept { return error_; }

    constexpr const E&& error() const&& noexcept { return std::move(error_); }
    constexpr E&& error() && noexcept { return std::move(error_); }

  private:
    E error_;
  };

  template <typename E>
  unexpected(E) -> unexpected<E>;

  namespace detail {
    template <typename T>
    struct is_unexpected : std::false_type {};

    template <typename E>
    struct is_unexpected<unexpected<E>> : std::true_type {};

    template <typename T>
    concept unexpected = is_unexpected<std::remove_cvref_t<T>>::value;
  }  // namespace detail

  template <typename T, typename E>
    requires(not detail::unexpected<T>)
  class expected;

  namespace detail {
    template <typename T>
    struct is_expected : std::false_type {};

    template <typename T, typename E>
    struct is_expected<edm::expected<T, E>> : std::true_type {};

    template <typename T>
    concept expected = is_expected<std::remove_cvref_t<T>>::value;
  }  // namespace detail

  /* Partial implementation of c++23 std::expected in c++20.
   *
   * Supported:
   *   - construction from a value (U&&) or an error (unexpected<G>), conditionally explicit;
   *   - copy and move construction, copy and move assignment;
   *   - has_value(), operator bool, operator*, operator->, value(), value_or(), error(), error_or();
   *   - edm::unexpected<E>, with error() and the unexpected(E) deduction guide.
   *
   * Different from std::expected:
   *   - value() throws std::logic_error instead of std::bad_expected_access<E>;
   *   - copy and move assignment require both T and E to be nothrow move constructible, instead of at least one,
   *     so that the std::variant storage never becomes valueless;
   *   - assigning a value or an error constructs a temporary edm::expected and moves it;
   *   - operator*, operator-> and error() assert on the wrong state, instead of undefined behaviour.
   *
   * Missing:
   *   - edm::expected<void, E> specialisation;
   *   - default construction, the std::in_place and std::unexpect constructors, emplace() and swap();
   *   - converting construction and assignment from edm::expected<U, G>;
   *   - comparison operators;
   *   - and_then(), or_else(), transform() and transform_error().
   */
  template <typename T, typename E>
    requires(not detail::unexpected<T>)
  class expected {
  public:
    template <typename U = std::remove_cv_t<T>>
      requires(not std::is_same_v<std::remove_cvref_t<U>, std::in_place_t> and
               not std::is_same_v<std::remove_cvref_t<U>, expected> and not detail::unexpected<U> and
               std::is_constructible_v<T, U> and
               (not std::is_same_v<std::remove_cv_t<T>, bool> or not detail::expected<U>))
    constexpr explicit(not std::is_convertible_v<U, T>) expected(U&& val)
        : storage_(std::in_place_index<0>, std::forward<U>(val)) {}

    template <typename G>
      requires std::is_constructible_v<E, G const&>
    constexpr explicit(not std::is_convertible_v<G const&, E>) expected(unexpected<G> const& err)
        : storage_(std::in_place_index<1>, err.error()) {}

    template <typename G>
      requires std::is_constructible_v<E, G>
    constexpr explicit(not std::is_convertible_v<G, E>) expected(unexpected<G>&& err)
        : storage_(std::in_place_index<1>, std::move(err).error()) {}

    constexpr expected(expected const&) = default;
    constexpr expected(expected&&) = default;

    // Stricter than std::expected, which requires only one of T and E to be nothrow move constructible:
    // with std::variant as storage, this guarantees that an assignment never leaves it valueless.
    constexpr expected& operator=(expected const&)
      requires(std::is_nothrow_move_constructible_v<T> and std::is_nothrow_move_constructible_v<E>)
    = default;
    constexpr expected& operator=(expected&&)
      requires(std::is_nothrow_move_constructible_v<T> and std::is_nothrow_move_constructible_v<E>)
    = default;

    constexpr const T* operator->() const noexcept {
      assert(has_value());
      return std::get_if<0>(&storage_);
    }
    constexpr T* operator->() noexcept {
      assert(has_value());
      return std::get_if<0>(&storage_);
    }

    constexpr const T& operator*() const& noexcept {
      assert(has_value());
      return *std::get_if<0>(&storage_);
    }
    constexpr T& operator*() & noexcept {
      assert(has_value());
      return *std::get_if<0>(&storage_);
    }

    constexpr const T&& operator*() const&& noexcept {
      assert(has_value());
      return std::move(*std::get_if<0>(&storage_));
    }
    constexpr T&& operator*() && noexcept {
      assert(has_value());
      return std::move(*std::get_if<0>(&storage_));
    }

    constexpr bool has_value() const noexcept { return std::holds_alternative<T>(storage_); }

    constexpr explicit operator bool() const noexcept { return has_value(); }

    constexpr T& value() & {
      check_value_();
      return std::get<T>(storage_);
    }

    constexpr const T& value() const& {
      check_value_();
      return std::get<T>(storage_);
    }

    constexpr T&& value() && {
      check_value_();
      return std::move(std::get<T>(storage_));
    }

    constexpr const T&& value() const&& {
      check_value_();
      return std::move(std::get<T>(storage_));
    }

    template <class U = std::remove_cv_t<T>>
    constexpr T value_or(U&& default_value) const& {
      static_assert(std::is_copy_constructible_v<T>);
      static_assert(std::is_convertible_v<U, T>);
      if (has_value()) {
        return **this;
      }
      return static_cast<T>(std::forward<U>(default_value));
    }

    template <class U = std::remove_cv_t<T>>
    constexpr T value_or(U&& default_value) && {
      static_assert(std::is_move_constructible_v<T>);
      static_assert(std::is_convertible_v<U, T>);
      if (has_value()) {
        return std::move(**this);
      }
      return static_cast<T>(std::forward<U>(default_value));
    }

    constexpr const E& error() const& noexcept {
      assert(not has_value());
      return std::get_if<1>(&storage_)->error();
    }
    constexpr E& error() & noexcept {
      assert(not has_value());
      return std::get_if<1>(&storage_)->error();
    }

    constexpr const E&& error() const&& noexcept {
      assert(not has_value());
      return std::move(std::get_if<1>(&storage_)->error());
    }
    constexpr E&& error() && noexcept {
      assert(not has_value());
      return std::move(std::get_if<1>(&storage_)->error());
    }

    template <class G = E>
    constexpr E error_or(G&& default_value) const& {
      static_assert(std::is_copy_constructible_v<E>);
      static_assert(std::is_convertible_v<G, E>);
      if (has_value()) {
        return std::forward<G>(default_value);
      }
      return error();
    }

    template <class G = E>
    constexpr E error_or(G&& default_value) && {
      static_assert(std::is_move_constructible_v<E>);
      static_assert(std::is_convertible_v<G, E>);
      if (has_value()) {
        return std::forward<G>(default_value);
      }
      return std::move(error());
    }

  private:
    std::variant<T, unexpected<E>> storage_;

    constexpr void check_value_() const {
      if (not has_value()) {
        throw std::logic_error("Access to an edm::expected object that contains an unexpected value.");
      }
    }
  };

}  // namespace edm

#endif  // FWCore_Utilities_expected_h
