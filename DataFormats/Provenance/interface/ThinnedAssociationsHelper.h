#ifndef DataFormats_Provenance_ThinnedAssociationsHelper_h
#define DataFormats_Provenance_ThinnedAssociationsHelper_h

/** \class edm::ThinnedAssociationsHelper
\author W. David Dagenhart, created 11 June 2014
*/

#include "DataFormats/Provenance/interface/BranchID.h"

#include <map>
#include <set>
#include <vector>

namespace edm {

  class BranchDescription;

  class ThinnedAssociationBranches {
  public:
    ThinnedAssociationBranches();
    ThinnedAssociationBranches(BranchID const&, BranchID const&, BranchID const&);

    BranchID const& parent() const { return parent_; }
    BranchID const& association() const { return association_; }
    BranchID const& thinned() const { return thinned_; }

    bool operator<(ThinnedAssociationBranches const& rhs) const { return parent_ < rhs.parent_; }

  private:
    BranchID parent_;
    BranchID association_;
    BranchID thinned_;
  };

  class ThinnedAssociationsHelper {
  public:

    ThinnedAssociationsHelper();

    std::vector<ThinnedAssociationBranches>::const_iterator begin() const;
    std::vector<ThinnedAssociationBranches>::const_iterator end() const;

    std::vector<ThinnedAssociationBranches>::const_iterator parentBegin(BranchID const&) const;
    std::vector<ThinnedAssociationBranches>::const_iterator parentEnd(BranchID const&) const;

    void addAssociation(BranchID const&, BranchID const&, BranchID const&);
    void addAssociation(ThinnedAssociationBranches const&);

    std::vector<std::pair<BranchID, ThinnedAssociationBranches const*> > associationToBranches() const;

    void sort();
    void clear() { vThinnedAssociationBranches_.clear(); }

    void selectAssociationProducts(std::vector<BranchDescription const*> const& associationDescriptions,
                                   std::set<BranchID> const& keptProductsInEvent,
                                   std::map<BranchID, bool>& keepAssociation) const;

    std::vector<ThinnedAssociationBranches> const& data() const { return vThinnedAssociationBranches_; }

    void requireMatch(ThinnedAssociationBranches const& input) const;

    void updateFromInput(ThinnedAssociationsHelper const&,
                         bool isSecondaryFile,
                         std::vector<BranchID> const& associationsFromSecondary);

    void updateFromParentProcess(ThinnedAssociationsHelper const& parentThinnedAssociationsHelper,
                                 std::map<BranchID, bool> const& keepAssociation,
                                 std::map<BranchID::value_type, BranchID::value_type> const& droppedBranchIDToKeptBranchID);

    void initAssociationsFromSecondary(std::vector<BranchID> const&,
                                       ThinnedAssociationsHelper const&);

  private:

    bool shouldKeepAssociation(BranchID const& association,
                               std::vector<std::pair<BranchID, ThinnedAssociationBranches const*> > const& associationToBranches,
                               std::set<BranchID>& branchesInRecursion,
                               std::set<BranchID> const& keptProductsInEvent,
                               std::map<BranchID, bool>& keepAssociation) const;


    std::vector<ThinnedAssociationBranches> vThinnedAssociationBranches_;
  };
}
#endif
