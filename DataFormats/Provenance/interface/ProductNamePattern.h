#ifndef DataFormats_Provenance_interface_ProductNamePattern_h
#define DataFormats_Provenance_interface_ProductNamePattern_h

#include <string>
#include <regex>
#include <vector>

#include "DataFormats/Provenance/interface/ProductDescription.h"

namespace edm {

  /* ProductNamePattern
   *
   * A ProductNamePattern is constructed from a string representing either a module label (e.g. "<module label>") or a
   * a branch name (e.g. "<product type>_<module label>_<instance name>_<process name>").
   *
   * A ProductNamePattern object can be compared with a ProductDescription object using the match() method:
   *
   *     productPattern.match(product)
   *
   * .
   * Glob expressions ("?" and "*") are supported in module labels and within the individual fields of branch names,
   * similar to an OutputModule's "keep" statements.
   * Use "*" to match all products of a given category.
   *
   * If a module label is used, it must not contain any underscores ("_"); the resulting ProductNamePattern will match all
   * the branches prodced by a module with the given label, including those with a non-empty instance names, and those
   * produced by the Transformer functionality (such as the implicitly copied-to-host products in case of Alpaka-based
   * modules).
   * If a branch name is used, all four fields must be present, separated by underscores; the resulting ProductNamePattern
   * will match the branches matching all four fields.
   * Only in this case and only for transient products, the module label may contain additional underscores.
   *
   * For example, in the case of products from an Alpaka-based producer running on a device
   *
   *     ProductNamePattern("module")
   *
   * would match all branches produced by "module", including the automatic host copy of its device products.
   * While
   *
   *     ProductNamePattern( "*DeviceProduct_module_*_*" )
   *
   * would match only the branches corresponding to the device products.
   */

  class ProductNamePattern {
  public:
    explicit ProductNamePattern(std::string const& label);

    bool match(edm::ProductDescription const& product) const;

  private:
    std::regex type_;
    std::regex moduleLabel_;
    std::regex productInstanceName_;
    std::regex processName_;
  };

  /* productPatterns
   *
   * Utility function to construct a vector<edm::ProductNamePattern> from a vector<std::string>.
   */
  std::vector<ProductNamePattern> productPatterns(std::vector<std::string> const& labels);

}  // namespace edm

#endif  // DataFormats_Provenance_interface_ProductNamePattern_h
