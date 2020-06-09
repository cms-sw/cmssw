#include "DetectorDescription/DDCMS/interface/Filter.h"
#include <functional>
#include <regex>
#include <iostream>

using namespace std;

namespace cms {
  namespace dd {

    bool isMatch(string_view node, string_view name) {
      if (!isRegex(name)) {
        return (name == node);
      } else {
        regex pattern({name.data(), name.size()});
        return regex_match(begin(node), end(node), pattern);
      }
    }

    bool compareEqual(string_view node, string_view name) { return (name == node); }

    bool compareEqual(string_view node, regex pattern) {
      return regex_match(std::string(node.data(), node.size()), pattern);
    }

    bool accepted(vector<std::regex> const& keys, string_view node) {
      return (find_if(begin(keys), end(keys), [&](const auto& n) -> bool { return compareEqual(node, n); }) !=
              end(keys));
    }

    bool isRegex(string_view input) {
      return (input.find(".") != std::string_view::npos) || (input.find("*") != std::string_view::npos);
    }

    string_view realTopName(string_view input) {
      string_view v = input;
      auto first = v.find_first_of("//");
      v.remove_prefix(min(first + 2, v.size()));
      return v;
    }

    vector<string_view> split(string_view str, const char* delims) {
      vector<string_view> ret;

      string_view::size_type start = 0;
      auto pos = str.find_first_of(delims, start);
      while (pos != string_view::npos) {
        if (pos != start) {
          ret.emplace_back(str.substr(start, pos - start));
        }
        start = pos + 1;
        pos = str.find_first_of(delims, start);
      }
      if (start < str.length())
        ret.emplace_back(str.substr(start, str.length() - start));
      return ret;
    }

    std::string_view noNamespace(std::string_view input) {
      std::string_view v = input;
      auto first = v.find_first_of(":");
      v.remove_prefix(std::min(first + 1, v.size()));
      return v;
    }
  }  // namespace dd
}  // namespace cms
