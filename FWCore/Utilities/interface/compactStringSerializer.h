#ifndef FWCore_Utilities_interface_compactStringSerializer_h
#define FWCore_Utilities_interface_compactStringSerializer_h

#include <cassert>
#include <iterator>
#include <numeric>
#include <ranges>
#include <string>
#include <string_view>

namespace edm::compactString {
  namespace detail {
    constexpr std::string_view kDelimiters = "\x1d\x1e";
    constexpr char kContainerDelimiter = kDelimiters[0];  // "group separator" in ASCII
    constexpr char kElementDelimiter = kDelimiters[1];    // "record separator" in ASCII

    void throwIfContainsDelimiters(std::string const& str);
  }  // namespace detail

  /**
   * Following three functions serialize a sequence of strings and containers of strings
   *
   * Each top-level string or container of strings is separated with kContainerDelimeter
   * In case of container of strings, each element is separated with kElementDelimeter
   * The serialized string will end with kContainerDelimeter and a null character
   *
   * The functions throw an exception if the serialized strings
   * contain any of the delimeter characters. The underlying string
   * operations may also throw exceptions.
   */
  inline std::string serialize(std::string arg) noexcept(false) {
    detail::throwIfContainsDelimiters(arg);
    arg += detail::kContainerDelimiter;
    return arg;
  }

  template <typename R>
    requires std::ranges::input_range<R> and std::is_same_v<std::ranges::range_value_t<R>, std::string>
  std::string serialize(R const& arg) noexcept(false) {
    std::string ret;

    if (not std::ranges::empty(arg)) {
      for (std::string const& elem : arg) {
        ret.reserve(ret.size() + elem.size() + 1);
        detail::throwIfContainsDelimiters(elem);
        ret += elem;
        ret += detail::kElementDelimiter;
      }
    }

    ret += detail::kContainerDelimiter;
    return ret;
  }

  template <typename T, typename... Args>
    requires(sizeof...(Args) >= 1)
  std::string serialize(T&& arg0, Args&&... args) noexcept(false) {
    return serialize(std::forward<T>(arg0)) + serialize(std::forward<Args>(args)...);
  }

  /**
   * Following three functions deserialize a string 'input' into a
   * sequence of strings and containers of strings
   *
   * The 'input' string is assumed to be serialized with the
   * serialize() functions above.
   *
   * The output arguments following the 'input' define the schema of
   * the deserialization.
   * - std::string& for strings
   * - output iterator for containers of strings (e.g. std::back_inserter(vector))
   *
   * Upon success, the return value is the position in `input` for the
   * next possible element (i.e. the position after the
   * kContainerDelimiter), that is also the number of characters
   * consumed by the deserializatiom..
   *
   * Upon failure, returns 0 to denote the beginning of `input`. The
   * output arguments may have been modified.
   *
   * The functions do not explicitly throw exceptions, but underlying
   * operations may throw exceptions.
   */
  inline std::string_view::size_type deserialize(std::string_view input, std::string& arg) {
    auto const pos = input.find_first_of(detail::kDelimiters);
    if (pos == std::string_view::npos or input[pos] != detail::kContainerDelimiter) {
      return 0;
    }
    arg = input.substr(0, pos);
    return pos + 1;  // skip delimiter
  }

  template <std::output_iterator<std::string> I>
  inline std::string_view::size_type deserialize(std::string_view input, I arg) {
    auto pos = input.find_first_of(detail::kDelimiters);
    // invalid input
    if (pos == std::string_view::npos) {
      return 0;
    }
    // no elements
    if (input[pos] == detail::kContainerDelimiter) {
      // invalid input for empty container
      if (pos != 0) {
        return 0;
      }
      // skip delimiter
      return pos + 1;
    }

    std::string_view::size_type prev = 0;
    while (pos != std::string_view::npos and input[pos] == detail::kElementDelimiter) {
      *arg = std::string(input.substr(prev, pos - prev));
      ++arg;
      prev = pos + 1;  //skip delimiter
      pos = input.find_first_of(detail::kDelimiters, prev);
    }

    // every container must end with kContainerDelimiter
    // reaching npos is an error
    if (pos == std::string_view::npos) {
      return 0;
    }
    assert(input[pos] == detail::kContainerDelimiter);

    return pos + 1;  // skip delimiter
  }

  template <typename T, typename... Args>
    requires(sizeof...(Args) >= 1)
  std::string_view::size_type deserialize(std::string_view input, T&& arg0, Args&&... args) {
    auto pos = deserialize(input, std::forward<T>(arg0));
    if (pos != 0) {
      auto const ret = deserialize(input.substr(pos), std::forward<Args>(args)...);
      pos = (ret == 0) ? 0 : pos + ret;
    }
    return pos;
  }
}  // namespace edm::compactString

#endif
