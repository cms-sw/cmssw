#include <algorithm>
#include <string>
#include <string_view>
#include <regex>
#include <vector>

#include <boost/algorithm/string/replace.hpp>

#include "DataFormats/Provenance/interface/ProductNamePattern.h"
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

  /* ProductNamePattern
   *
   * A ProductNamePattern is constructed from a string representing either a module label (e.g. "<module label>") or a
   * a branch name (e.g. "<product type>_<module label>_<instance name>_<process name>").
   *
   * For transient products, the module label may contain additional underscores.
   *
   * See DataFormats/Provenance/interface/ProductNamePattern.h for more details.
   */
  ProductNamePattern::ProductNamePattern(std::string const& label) {
    static constexpr char kSeparator = '_';
    static constexpr std::string_view kWildcard{"*"};
    static const std::regex kAny{".*"};
    static const std::regex kFields{"([a-zA-Z0-9*]+)_([a-zA-Z0-9_*]+)_([a-zA-Z0-9*]*)_([a-zA-Z0-9*]+)"};

    // empty label
    if (label.empty()) {
      throw edm::Exception(edm::errors::Configuration) << "Invalid module label or branch name: \"" << label << "\"";
    }

    // wildcard
    if (label == kWildcard) {
      type_ = kAny;
      moduleLabel_ = kAny;
      productInstanceName_ = kAny;
      processName_ = kAny;
      return;
    }

    int underscores = std::count(label.begin(), label.end(), kSeparator);
    if (underscores == 0) {
      // Convert the module label into a regular expression.
      type_ = kAny;
      moduleLabel_ = glob_to_regex(label);
      productInstanceName_ = kAny;
      processName_ = kAny;
      return;
    }

    if (underscores >= 3) {
      // Split the branch name into <product type>_<module label>_<instance name>_<process name>
      // and convert the glob expressions into regular expressions.
      // Note that:
      //   - the <instance name> may be empty;
      //   - for non-persistable branches, <module label> may contain additional underscores.
      std::smatch fields;
      if (std::regex_match(label, fields, kFields)) {
        type_ = glob_to_regex(fields[1]);
        moduleLabel_ = glob_to_regex(fields[2]);
        productInstanceName_ = glob_to_regex(fields[3]);
        processName_ = glob_to_regex(fields[4]);
        return;
      }
    }

    // Invalid input.
    throw edm::Exception(edm::errors::Configuration) << "Invalid module label or branch name: \"" << label << "\"";
  }

  /* ProductNamePattern::match
   *
   * Compare a ProductNamePattern object with a ProductDescription object.
   */
  bool ProductNamePattern::match(edm::ProductDescription const& product) const {
    return (std::regex_match(product.friendlyClassName(), type_) and
            std::regex_match(product.moduleLabel(), moduleLabel_) and
            std::regex_match(product.productInstanceName(), productInstanceName_) and
            std::regex_match(product.processName(), processName_));
  }

  /* productPatterns
   *
   * Utility function to construct a vector<edm::ProductNamePattern> from a vector<std::string>.
   */
  std::vector<ProductNamePattern> productPatterns(std::vector<std::string> const& labels) {
    std::vector<ProductNamePattern> patterns;
    patterns.reserve(labels.size());
    for (auto const& label : labels)
      patterns.emplace_back(label);
    return patterns;
  }

}  // namespace edm
