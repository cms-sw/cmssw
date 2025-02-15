#ifndef DataFormats_Provenance_interface_BranchPattern_h
#define DataFormats_Provenance_interface_BranchPattern_h

#include <string>
#include <regex>
#include <vector>

#include "DataFormats/Provenance/interface/ProductDescription.h"

namespace edm {

  /* BranchPattern
   *
   * A BranchPattern is constructed from a string representing either a module label (e.g. "<module label>") or a
   * a branch name (e.g. "<product type>_<module label>_<instance name>_<process name>").
   *
   * A BranchPattern object can be compared with a ProductDescription object using the match() method:
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
    explicit BranchPattern(std::string const& label);

    bool match(edm::ProductDescription const& branch) const;

  private:
    std::regex type_;
    std::regex moduleLabel_;
    std::regex productInstanceName_;
    std::regex processName_;
  };

  /* branchPatterns
   *
   * Utility function to construct a vector<edm::BranchPattern> from a vector<std::string>.
   */
  std::vector<BranchPattern> branchPatterns(std::vector<std::string> const& labels);

}  // namespace edm

#endif  // DataFormats_Provenance_interface_BranchPattern_h
