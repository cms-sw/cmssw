#ifndef FWCore_Framework_ProductSelector_h
#define FWCore_Framework_ProductSelector_h

//////////////////////////////////////////////////////////////////////
//
// Class ProductSelector. Class for user to select specific products in event.
//
// Author: Bill Tanenbaum, Marc Paterno
//
//////////////////////////////////////////////////////////////////////

#include <iosfwd>
#include <string>
#include <vector>

namespace edm {
  class BranchDescription;
  class ProductSelectorRules;
  class ParameterSet;

  class ProductSelector {
  public:
    ProductSelector();

    // N.B.: we assume there are not null pointers in the vector allBranches.
    void initialize(ProductSelectorRules const& rules,
		    std::vector<BranchDescription const*> const& branchDescriptions);

    bool selected(BranchDescription const& desc) const;

    // Printout intended for debugging purposes.
    void print(std::ostream& os) const;

    bool initialized() const {return initialized_;}

  private:

    // We keep a sorted collection of branch names, indicating the
    // products which are to be selected.

    // TODO: See if we can keep pointer to (const) BranchDescriptions,
    // so that we can do pointer comparison rather than string
    // comparison. This will work if the BranchDescription we are
    // given in the 'selected' member function is one of the instances
    // that are managed by the ProductRegistry used to initialize the
    // entity that contains this ProductSelector.
    std::vector<std::string> productsToSelect_;
    bool initialized_;
  };

  std::ostream&
  operator<< (std::ostream& os, const ProductSelector& gs);

} // namespace edm



#endif
