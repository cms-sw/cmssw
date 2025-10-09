#include "catch2/catch_all.hpp"
namespace trigger_results_based_event_selector_utils {
  // Implemented in TriggerResultsBasedEventSelector.cc
  void remove_whitespace(std::string& s);

  typedef std::pair<std::string, std::string> parsed_path_spec_t;
  void parse_path_spec(std::string const& path_spec, parsed_path_spec_t& output);
}  // namespace trigger_results_based_event_selector_utils

using namespace trigger_results_based_event_selector_utils;
TEST_CASE("test TriggerResultsBasedEventSelector utilities", "[TriggerResultsBasedEventSelectorUtils]") {
  SECTION("remove whitespace") {
    std::string a("noblanks");
    std::string b("\t   no   blanks    \t");

    remove_whitespace(b);
    CHECK(a == b);
  }
  SECTION("test_parse_path_spec") {
    std::vector<std::string> paths;
    paths.push_back("a:p1");
    paths.push_back("b:p2");
    paths.push_back("  c");
    paths.push_back("ddd\t:p3");
    paths.push_back("eee:  p4  ");

    std::vector<parsed_path_spec_t> parsed(paths.size());
    for (size_t i = 0; i < paths.size(); ++i) {
      parse_path_spec(paths[i], parsed[i]);
    }

    CHECK(parsed[0].first == "a");
    CHECK(parsed[0].second == "p1");
    CHECK(parsed[1].first == "b");
    CHECK(parsed[1].second == "p2");
    CHECK(parsed[2].first == "c");
    CHECK(parsed[2].second.empty());
    CHECK(parsed[3].first == "ddd");
    CHECK(parsed[3].second == "p3");
    CHECK(parsed[4].first == "eee");
    CHECK(parsed[4].second == "p4");
  }
}
