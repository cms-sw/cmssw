#ifndef DataFormats_Provenance_interface_BranchPattern_h
#define DataFormats_Provenance_interface_BranchPattern_h

#include <algorithm>
#include <string>
#include <string_view>
#include <regex>
#include <vector>

#include <boost/algorithm/string/replace.hpp>

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {

  /* BranchPattern
   *
   * A BranchPattern is constructed from a string representing either a module label (e.g. "<module label>") or a
   * a branch name (e.g. "<product type>_<module label>_<instance name>_<process name>").
   *
   * A BranchPattern object can be compared with a BranchDescription object using the match() method:
   *
   *     branchPattern.match(branch)
   *
   * .
   * Glob expressions ("?" and "*") are supported in module labels and within the individual fields of branch names,
   * similar to an OutputModule's "keep" statements.
   * Use "*" to match all products of a given category.
   *
   * If a module label is used, it must not contain any underscores ("_"); the resulting BranchPattern will match all
   * the branches prodced by a module with the given label, including those with a non-empty instance names, and those
   * produced by the Transformer functionality (such as the implicitly copied-to-host products in case of Alpaka-based
   * modules).
   * If a branch name is used, all four fields must be present, separated by underscores; the resulting BranchPattern
   * will match the branches matching all four fields.
   *
   * For example, in the case of products from an Alpaka-based producer running on a device
   *
   *     BranchPattern("module")
   *
   * would match all branches produced by "module", including the automatic host copy of its device products.
   * While
   *
   *     BranchPattern( "*DeviceProduct_module_*_*" )
   *
   * would match only the branches corresponding to the device products.
   */

  class BranchPattern {
  public:
    explicit BranchPattern(std::string const& label) {
      static const char kSeparator = '_';
      static const std::string_view kWildcard{"*"};
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
        // convert the module label into a regular expression
        type_ = kAny;
        moduleLabel_ = glob_to_regex(label);
        productInstanceName_ = kAny;
        processName_ = kAny;
      } else if (fields == 4) {
        // split the branch name into <product type>_<module label>_<instance name>_<process name>
        // and convert the glob expressions into regular expressions
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
        // invalid input
        throw edm::Exception(edm::errors::Configuration) << "Invalid module label or branch name: \"" << label << "\"";
      }
    }

    bool match(edm::BranchDescription const& branch) const {
      return (std::regex_match(branch.friendlyClassName(), type_) and
              std::regex_match(branch.moduleLabel(), moduleLabel_) and
              std::regex_match(branch.productInstanceName(), productInstanceName_) and
              std::regex_match(branch.processName(), processName_));
    }

  private:
    static std::regex glob_to_regex(std::string pattern) {
      boost::replace_all(pattern, "*", ".*");
      boost::replace_all(pattern, "?", ".");
      return std::regex(pattern);
    }

    std::regex type_;
    std::regex moduleLabel_;
    std::regex productInstanceName_;
    std::regex processName_;
  };

  inline std::vector<BranchPattern> branchPatterns(std::vector<std::string> const& labels) {
    std::vector<BranchPattern> patterns;
    patterns.reserve(labels.size());
    for (auto const& label : labels)
      patterns.emplace_back(label);
    return patterns;
  }

}  // namespace edm

#endif  // DataFormats_Provenance_interface_BranchPattern_h
