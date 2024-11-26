#include <algorithm>
#include <string>
#include <string_view>
#include <regex>
#include <vector>

#include <boost/algorithm/string/replace.hpp>

#include "DataFormats/Provenance/interface/BranchPattern.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {

  /* glob_to_regex
   *
   * Utility function to convert a shell-like glob expression to a regex.
   */
  static std::regex glob_to_regex(std::string pattern) {
    boost::replace_all(pattern, "*", ".*");
    boost::replace_all(pattern, "?", ".");
    return std::regex(pattern);
  }

  /* BranchPattern
   *
   * A BranchPattern is constructed from a string representing either a module label (e.g. "<module label>") or a
   * a branch name (e.g. "<product type>_<module label>_<instance name>_<process name>").
   *
   * See DataFormats/Provenance/interface/BranchPattern.h for more details.
   */
  BranchPattern::BranchPattern(std::string const& label) {
    static constexpr char kSeparator = '_';
    static constexpr std::string_view kWildcard{"*"};
    static const std::regex kAny{".*"};

    // wildcard
    if (label == kWildcard) {
      type_ = kAny;
      moduleLabel_ = kAny;
      productInstanceName_ = kAny;
      processName_ = kAny;
      return;
    }

    int fields = std::count(label.begin(), label.end(), kSeparator) + 1;
    if (fields == 1) {
      // Convert the module label into a regular expression.
      type_ = kAny;
      moduleLabel_ = glob_to_regex(label);
      productInstanceName_ = kAny;
      processName_ = kAny;
    } else if (fields == 4) {
      // Split the branch name into <product type>_<module label>_<instance name>_<process name>
      // and convert the glob expressions into regular expressions.
      // FIXME <module label> is the only field that may contain additional underscores, for non-persistable branches.
      size_t first = 0, last = 0;
      last = label.find(kSeparator, first);
      type_ = glob_to_regex(label.substr(first, last - first));
      first = last + 1;
      last = label.find(kSeparator, first);
      moduleLabel_ = glob_to_regex(label.substr(first, last - first));
      first = last + 1;
      last = label.find(kSeparator, first);
      productInstanceName_ = glob_to_regex(label.substr(first, last - first));
      first = last + 1;
      last = label.find(kSeparator, first);
      processName_ = glob_to_regex(label.substr(first, last - first));
    } else {
      // Invalid input.
      throw edm::Exception(edm::errors::Configuration) << "Invalid module label or branch name: \"" << label << "\"";
    }
  }

  /* BranchPattern::match
   *
   * Compare a BranchPattern object with a ProductDescription object.
   */
  bool BranchPattern::match(edm::ProductDescription const& branch) const {
    return (std::regex_match(branch.friendlyClassName(), type_) and
            std::regex_match(branch.moduleLabel(), moduleLabel_) and
            std::regex_match(branch.productInstanceName(), productInstanceName_) and
            std::regex_match(branch.processName(), processName_));
  }

  /* branchPatterns
   *
   * Utility function to construct a vector<edm::BranchPattern> from a vector<std::string>.
   */
  std::vector<BranchPattern> branchPatterns(std::vector<std::string> const& labels) {
    std::vector<BranchPattern> patterns;
    patterns.reserve(labels.size());
    for (auto const& label : labels)
      patterns.emplace_back(label);
    return patterns;
  }

}  // namespace edm
