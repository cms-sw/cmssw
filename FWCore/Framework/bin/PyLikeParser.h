#ifndef FWCore_Framework_bin_PyLikeParser_h
#define FWCore_Framework_bin_PyLikeParser_h

#include <cerrno>
#include <concepts>
#include <cstdlib>
#include <cstdlib>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace edm {

  class PyLikeParser {
  public:
    explicit PyLikeParser(std::string_view input) : data_(input) {}

    template <typename T>
    T parse() {
      T value;

      // skip any leading spaces, and check that the input is not empty
      skip_whitespace();
      check_for_content();

      if constexpr (std::is_same_v<T, bool>)
        value = parse_bool_();
      else if constexpr (std::is_same_v<T, int>)
        value = parse_int32_();
      else if constexpr (std::is_same_v<T, unsigned int>)
        value = parse_uint32_();
      else if constexpr (std::is_same_v<T, long long>)
        value = parse_int64_();
      else if constexpr (std::is_same_v<T, unsigned long long>)
        value = parse_uint64_();
      else if constexpr (std::is_same_v<T, float>)
        value = parse_float_();
      else if constexpr (std::is_same_v<T, double>)
        value = parse_double_();
      else if constexpr (std::is_same_v<T, std::string>)
        value = parse_string_();
      else
        static_assert("Unsupported type");

      // skip any trailing spaces, and check that the whole string has been parsed
      skip_whitespace();
      check_no_more_content();

      return value;
    }

    template <typename T>
    std::vector<T> parse_vector() {
      std::vector<T> value;
      do {
        skip_whitespace();
        // a list can be empty, or can have a trailing comma
        if (at_end()) {
          return value;
        }

        if constexpr (std::is_same_v<T, bool>)
          value.push_back(parse_bool_());
        else if constexpr (std::is_same_v<T, int>)
          value.push_back(parse_int32_());
        else if constexpr (std::is_same_v<T, unsigned int>)
          value.push_back(parse_uint32_());
        else if constexpr (std::is_same_v<T, long long>)
          value.push_back(parse_int64_());
        else if constexpr (std::is_same_v<T, unsigned long long>)
          value.push_back(parse_uint64_());
        else if constexpr (std::is_same_v<T, float>)
          value.push_back(parse_float_());
        else if constexpr (std::is_same_v<T, double>)
          value.push_back(parse_double_());
        else if constexpr (std::is_same_v<T, std::string>)
          value.push_back(parse_string_());
        else
          static_assert("Unsupported type");
      } while (consume_comma());

      check_no_more_content();
      return value;
    }

  private:
    bool parse_bool_() {
      skip_whitespace();
      if (data_.substr(pos_, 4) == "True") {
        pos_ += 4;
        return true;
      }
      if (data_.substr(pos_, 5) == "False") {
        pos_ += 5;
        return false;
      }
      fail("Invalid bool argument");
    }

    int parse_int32_() {
      const char* start_ptr = data_.data() + pos_;
      char* end_ptr = nullptr;
      errno = 0;
      long value = std::strtol(start_ptr, &end_ptr, 0);
      if (end_ptr == start_ptr || errno == ERANGE)
        fail("Invalid int32 value");
      if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
        fail("Invalid int32 value");
      pos_ += (end_ptr - start_ptr);
      return static_cast<int>(value);
    }

    unsigned int parse_uint32_() {
      const char* start_ptr = data_.data() + pos_;
      char* end_ptr = nullptr;
      errno = 0;
      unsigned long value = std::strtoul(start_ptr, &end_ptr, 0);
      if (end_ptr == start_ptr || errno == ERANGE)
        fail("Invalid uint32 value");
      if (value > std::numeric_limits<unsigned int>::max())
        fail("Invalid uint32 value");
      pos_ += (end_ptr - start_ptr);
      return static_cast<unsigned int>(value);
    }

    long long parse_int64_() {
      const char* start_ptr = data_.data() + pos_;
      char* end_ptr = nullptr;
      errno = 0;
      long long value = std::strtoll(start_ptr, &end_ptr, 0);
      if (end_ptr == start_ptr || errno == ERANGE)
        fail("Invalid int64 value");
      pos_ += (end_ptr - start_ptr);
      return value;
    }

    unsigned long long parse_uint64_() {
      const char* start_ptr = data_.data() + pos_;
      char* end_ptr = nullptr;
      errno = 0;
      unsigned long long value = std::strtoull(start_ptr, &end_ptr, 0);
      if (end_ptr == start_ptr || errno == ERANGE)
        fail("Invalid uint64 value");
      pos_ += (end_ptr - start_ptr);
      return value;
    }

    float parse_float_() {
      const char* start_ptr = data_.data() + pos_;
      char* end_ptr = nullptr;
      errno = 0;
      float value = std::strtof(start_ptr, &end_ptr);
      if (end_ptr == start_ptr || errno == ERANGE)
        fail("Invalid float argument");
      pos_ += (end_ptr - start_ptr);
      return value;
    }

    double parse_double_() {
      const char* start_ptr = data_.data() + pos_;
      char* end_ptr = nullptr;
      errno = 0;
      double value = std::strtod(start_ptr, &end_ptr);
      if (end_ptr == start_ptr || errno == ERANGE)
        fail("Invalid double argument");
      pos_ += (end_ptr - start_ptr);
      return value;
    }

    std::string parse_string_() {
      skip_whitespace();

      char quote = data_[pos_];
      if (quote != '"' && quote != '\'')
        fail("Invalid string argument");
      ++pos_;

      std::string value;
      // check against size() - 1 to leave space for the closing quote
      while (pos_ < data_.size() - 1 && data_[pos_] != quote) {
        // interpret common escape sequences: '\n', '\t', '\'', '\"', and '\\'
        if (data_[pos_] == '\\') {
          if (pos_ == data_.size() - 2) {
            // escape at the end of the string, missing closing quote
            fail("Missing closing quote");
          }
          ++pos_;
          // convert \n to newline
          if (data_[pos_] == 'n') {
            value.push_back('\n');
            ++pos_;
            continue;
          }
          // convert \t to tab
          if (data_[pos_] == 't') {
            value.push_back('\t');
            ++pos_;
            continue;
          }
          // any other escape character is converted to the character itself
        }
        value.push_back(data_[pos_]);
        ++pos_;
      }

      if (data_[pos_] != quote) {
        // missing closing quote
        fail("Missing closing quote");
      }
      ++pos_;
      return value;
    }

    void skip_whitespace() {
      while (not at_end() and std::isspace(static_cast<unsigned char>(data_[pos_]))) {
        pos_++;
      }
    }

    bool at_end() const {
      if (pos_ >= data_.size()) {
        return true;
      } else {
        return false;
      }
    }

    void check_for_content() const {
      if (at_end())
        fail("Missing argument");
    }

    void check_no_more_content() const {
      if (not at_end())
        fail("Invalid argument");
    }

    bool consume_comma() {
      skip_whitespace();
      if (pos_ < data_.size() && data_[pos_] == ',') {
        pos_++;
        return true;
      }
      return false;
    }

    [[noreturn]] void fail(const char* arg = "") const {
      throw std::runtime_error(std::string(arg) + std::string(": \"") + std::string(data_) + std::string("\""));
    }

    std::string_view data_;
    size_t pos_ = 0;
  };

}  // namespace edm

#endif  // FWCore_Framework_bin_PyLikeParser_h
