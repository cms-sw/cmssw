#ifndef FWCore_Framework_ProductSelector_h
#define FWCore_Framework_ProductSelector_h

//////////////////////////////////////////////////////////////////////
//
// Class ProductSelector. Class for user to select specific products in event.
//
// Author: Bill Tanenbaum, Marc Paterno
//
//////////////////////////////////////////////////////////////////////

#include "DataFormats/Provenance/interface/BranchID.h"

#include <iosfwd>
#include <map>
#include <string>
#include <vector>

namespace edm {
  class ProductDescription;
  class BranchID;
  class ProductRegistry;
  class ProductSelectorRules;
  class ParameterSet;

  class ProductSelector {
  public:
    ProductSelector();

    // N.B.: we assume there are not null pointers in the vector allBranches.
    void initialize(ProductSelectorRules const& rules, std::vector<ProductDescription const*> const& productDescriptions);

    bool selected(ProductDescription const& desc) const;

    // Printout intended for debugging purposes.
    void print(std::ostream& os) const;

    bool initialized() const { return initialized_; }

    static void checkForDuplicateKeptBranch(ProductDescription const& desc,
                                            std::map<BranchID, ProductDescription const*>& trueBranchIDToKeptBranchDesc);

    static void fillDroppedToKept(ProductRegistry const& preg,
                                  std::map<BranchID, ProductDescription const*> const& trueBranchIDToKeptBranchDesc,
                                  std::map<BranchID::value_type, BranchID::value_type>& droppedBranchIDToKeptBranchID_);

  private:
    // We keep a sorted collection of branch names, indicating the
    // products which are to be selected.

    // TODO: See if we can keep pointer to (const) ProductDescriptions,
    // so that we can do pointer comparison rather than string
    // comparison. This will work if the ProductDescription we are
    // given in the 'selected' member function is one of the instances
    // that are managed by the ProductRegistry used to initialize the
    // entity that contains this ProductSelector.
    std::vector<std::string> productsToSelect_;
    bool initialized_;
  };

  std::ostream& operator<<(std::ostream& os, const ProductSelector& gs);

}  // namespace edm

#endif
