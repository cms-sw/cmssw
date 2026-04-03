#include <stdexcept>
#include <variant>

namespace edm {

  // A simple wrapper to distinguish the error type from the value type
  template <typename E>
  struct unexpected {
    E error;

    constexpr explicit unexpected(E const& e) : error(e) {}
    constexpr explicit unexpected(E&& e) : error(std::move(e)) {}
  };

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
  class expected {
  public:
    constexpr expected(const T& val) : storage_(val) {}
    constexpr expected(T&& val) : storage_(std::move(val)) {}

    constexpr expected(unexpected<E> const& err) : storage_(err) {}
    constexpr expected(unexpected<E>&& err) : storage_(std::move(err)) {}

    constexpr const T* operator->() const noexcept { return &std::get<T>(storage_); }
    constexpr T* operator->() noexcept { return &std::get<T>(storage_); }

    constexpr const T& operator*() const& noexcept { return std::get<T>(storage_); }
    constexpr T& operator*() & noexcept { return std::get<T>(storage_); }

    constexpr const T&& operator*() const&& noexcept { return std::move(std::get<T>(storage_)); }
    constexpr T&& operator*() && noexcept { return std::move(std::get<T>(storage_)); }

    constexpr bool has_value() const noexcept { return std::holds_alternative<T>(storage_); }

    explicit operator bool() const noexcept { return has_value(); }

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
      return has_value() ? std::get<T>(storage_) : static_cast<T>(std::forward<U>(default_value));
    }

    template <class U = std::remove_cv_t<T>>
    constexpr T value_or(U&& default_value) && {
      return has_value() ? std::move(std::get<T>(storage_)) : static_cast<T>(std::forward<U>(default_value));
    }

    constexpr const E& error() const& noexcept { return std::get<unexpected<E>>(storage_).error; }
    constexpr E& error() & noexcept { return std::get<unexpected<E>>(storage_).error; }

    constexpr const E&& error() const&& noexcept { return std::move(std::get<unexpected<E>>(storage_).error); }
    constexpr E&& error() && noexcept { return std::move(std::get<unexpected<E>>(storage_).error); }

    template <class G = E>
    constexpr E error_or(G&& default_value) const& {
      return has_value() ? std::forward<G>(default_value) : error();
    }

    template <class G = E>
    constexpr E error_or(G&& default_value) && {
      return has_value() ? std::forward<G>(default_value) : std::move(error());
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
