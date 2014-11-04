#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <algorithm>

namespace edm {

  ThinnedAssociationBranches::ThinnedAssociationBranches() { }

  ThinnedAssociationBranches::ThinnedAssociationBranches(BranchID const& parent,
                                                         BranchID const& association,
                                                         BranchID const& thinned) :
    parent_(parent),
    association_(association),
    thinned_(thinned) {
  }

  ThinnedAssociationsHelper::ThinnedAssociationsHelper() {
  }

  std::vector<ThinnedAssociationBranches>::const_iterator
  ThinnedAssociationsHelper::begin() const {
    return vThinnedAssociationBranches_.begin();
  }

  std::vector<ThinnedAssociationBranches>::const_iterator
  ThinnedAssociationsHelper::end() const {
    return vThinnedAssociationBranches_.end();
  }

  std::vector<ThinnedAssociationBranches>::const_iterator
  ThinnedAssociationsHelper::parentBegin(BranchID const& parent) const {
    ThinnedAssociationBranches target(parent, BranchID(), BranchID());
    return std::lower_bound(vThinnedAssociationBranches_.begin(), vThinnedAssociationBranches_.end(), target,
                            [](ThinnedAssociationBranches const& x,
                               ThinnedAssociationBranches const& y)
                            { return x.parent() < y.parent(); });
  }

  std::vector<ThinnedAssociationBranches>::const_iterator
  ThinnedAssociationsHelper:: parentEnd(BranchID const& parent) const {
    ThinnedAssociationBranches target(parent, BranchID(), BranchID());
    return std::upper_bound(vThinnedAssociationBranches_.begin(), vThinnedAssociationBranches_.end(), target,
                            [](ThinnedAssociationBranches const& x,
                               ThinnedAssociationBranches const& y)
                            { return x.parent() < y.parent(); });
  }

  void ThinnedAssociationsHelper::sort() {
    std::sort(vThinnedAssociationBranches_.begin(), vThinnedAssociationBranches_.end(),
              [](ThinnedAssociationBranches const& x,
                 ThinnedAssociationBranches const& y)
              { return x.parent() < y.parent() ? true : y.parent() < x.parent() ? false : x.association() < y.association(); });
  }

  void ThinnedAssociationsHelper::addAssociation(BranchID const& parent,
                                                 BranchID const& association,
                                                 BranchID const& thinned) {
    vThinnedAssociationBranches_.push_back(ThinnedAssociationBranches(parent, association, thinned));
  }

  void ThinnedAssociationsHelper::addAssociation(ThinnedAssociationBranches const& branches) {
    vThinnedAssociationBranches_.push_back(branches);
  }

  std::vector<std::pair<BranchID, ThinnedAssociationBranches const*> >
  ThinnedAssociationsHelper::associationToBranches() const {
    std::vector<std::pair<BranchID, ThinnedAssociationBranches const*> > temp;
    for(auto const& item : vThinnedAssociationBranches_) {
      temp.push_back(std::make_pair(item.association(), &item));
    }
    std::sort(temp.begin(), temp.end(), [](std::pair<BranchID, ThinnedAssociationBranches const*> const& x,
                                           std::pair<BranchID, ThinnedAssociationBranches const*> const& y)
                                          { return x.first < y.first; });
    return temp;
  }

  void
  ThinnedAssociationsHelper::
  selectAssociationProducts(std::vector<BranchDescription const*> const& associationDescriptions,
                            std::set<BranchID> const& keptProductsInEvent,
                            std::map<BranchID, bool>& keepAssociation) const {

    keepAssociation.clear();
    // Copy the elements in vThinnedAssociationBranches_ into a vector sorted on
    // the association BranchID so we can do searches on that BranchID faster.
    std::vector<std::pair<BranchID, ThinnedAssociationBranches const*> > assocToBranches =
      associationToBranches();

    for(auto association : associationDescriptions) {
      if(association->isAlias()) { // There is no reason to configure an association product with an EDAlias (ignore and drop them if they exist)
        keepAssociation.insert(std::make_pair(association->branchID(), false));
      } else {
        std::set<BranchID> branchesInRecursion;
        shouldKeepAssociation(association->branchID(),
                              assocToBranches,
                              branchesInRecursion,
                              keptProductsInEvent,
                              keepAssociation);
      }
    }
  }

  bool ThinnedAssociationsHelper::shouldKeepAssociation(
    BranchID const& association,
    std::vector<std::pair<BranchID, ThinnedAssociationBranches const*> > const& associationToBranches,
    std::set<BranchID>& branchesInRecursion,
    std::set<BranchID> const& keptProductsInEvent,
    std::map<BranchID, bool>& keepAssociation) const {

    // If we already decided to keep or drop this one, then
    // return the same decision.
    auto decision = keepAssociation.find(association);
    if(decision != keepAssociation.end()) {
      return decision->second;
    }

    // Be careful not to fall into an infinite loop because
    // of a circular recursion.
    if(!branchesInRecursion.insert(association).second) {
      return false;
    }

    // If the thinned collection is being kept then keep the association
    auto branches = std::lower_bound(associationToBranches.begin(), associationToBranches.end(),
                                     std::make_pair(association, static_cast<ThinnedAssociationBranches const*>(nullptr)),
                                      [](std::pair<BranchID, ThinnedAssociationBranches const*> const& x,
                                         std::pair<BranchID, ThinnedAssociationBranches const*> const& y)
                                        { return x.first < y.first; });
    // This should never happen
    if(branches == associationToBranches.end() || branches->first != association) {
      throw edm::Exception(errors::LogicError, "ThinnedAssociationHelper::shouldKeepAssociation could not find branches information, contact Framework developers");
    }
    BranchID const& thinnedCollection = branches->second->thinned();
    if(keptProductsInEvent.find(thinnedCollection) != keptProductsInEvent.end()) {
      keepAssociation.insert(std::make_pair(association , true));
      return true;
    }
    // otherwise loop over any associations where the thinned collection
    // is also a parent collection and recursively examine those to see
    // if their thinned collections are being kept.
    auto iterEnd = parentEnd(thinnedCollection);
    for(auto match = parentBegin(thinnedCollection); match != iterEnd; ++match) {
      if(shouldKeepAssociation(match->association(),
                               associationToBranches,
                               branchesInRecursion,
                               keptProductsInEvent,
                               keepAssociation)) {
        keepAssociation.insert(std::make_pair(association , true));
        return true;
      }
    }
    // drop the association
    keepAssociation.insert(std::make_pair(association , false));
    return false;
  }

  void ThinnedAssociationsHelper::requireMatch(ThinnedAssociationBranches const& input) const {
    bool foundMatch = false;
    for(auto entry = parentBegin(input.parent()), iEnd = parentEnd(input.parent());
        entry != iEnd; ++entry) {
      if(entry->association() == input.association() &&
         entry->thinned() == input.thinned()) {
        foundMatch = true;
        break;
      }
    }
    if(!foundMatch) {
      throw edm::Exception(errors::MismatchedInputFiles, "ThinnedAssociationHelper::requireMatch, Illegal attempt to merge files with different ThinnedAssociations");
    }
  }

  void ThinnedAssociationsHelper::updateFromInput(ThinnedAssociationsHelper const& helper, bool isSecondaryFile,
                                                  std::vector<BranchID> const& associationsFromSecondary) {
    if(!isSecondaryFile) {
      if(vThinnedAssociationBranches_.empty()) {
        vThinnedAssociationBranches_ = helper.data();
        return;
      }
      std::vector<ThinnedAssociationBranches> const& inputData = helper.data();
      for (auto const& inputEntry : inputData) {
        requireMatch(inputEntry);
      }
    } else { // Input is from a secondary file

      if(associationsFromSecondary.empty()) return;

      std::vector<std::pair<BranchID, ThinnedAssociationBranches const*> > assocToBranches = helper.associationToBranches();

      for(BranchID const& association : associationsFromSecondary) {

        auto branches = std::lower_bound(assocToBranches.begin(), assocToBranches.end(),
                                         std::make_pair(association, static_cast<ThinnedAssociationBranches const*>(nullptr)),
                                          [](std::pair<BranchID, ThinnedAssociationBranches const*> const& x,
                                             std::pair<BranchID, ThinnedAssociationBranches const*> const& y)
                                            { return x.first < y.first; });
        // This should never happen
        if(branches == assocToBranches.end() || branches->first != association) {
          throw edm::Exception(errors::LogicError,
                             "ThinnedAssociationHelper::initAssociationsFromSecondary could not find branches information, contact Framework developers");
        }
        requireMatch(*(branches->second));
      }
    }
  }

  void
  ThinnedAssociationsHelper::updateFromParentProcess(ThinnedAssociationsHelper const& parentThinnedAssociationsHelper,
                                                     std::map<BranchID, bool> const& keepAssociation,
                                                     std::map<BranchID::value_type, BranchID::value_type> const& droppedBranchIDToKeptBranchID) {
    clear();
    for(auto const& associationBranches : parentThinnedAssociationsHelper.data()) {
      auto keep = keepAssociation.find(associationBranches.association());
      if(keep != keepAssociation.end() && keep->second) {
        BranchID parent = associationBranches.parent();
        auto iter = droppedBranchIDToKeptBranchID.find(parent.id());
        if(iter != droppedBranchIDToKeptBranchID.end()) {
          parent = BranchID(iter->second);
        }
        BranchID thinned = associationBranches.thinned();
        iter = droppedBranchIDToKeptBranchID.find(thinned.id());
        if(iter != droppedBranchIDToKeptBranchID.end()) {
          thinned = BranchID(iter->second);
        }
        addAssociation(parent, associationBranches.association(), thinned);
      }
    }
    sort();
  }

  void
  ThinnedAssociationsHelper::initAssociationsFromSecondary(std::vector<BranchID> const& associationsFromSecondary,
                                                           ThinnedAssociationsHelper const& fileAssociationsHelper) {
    if(associationsFromSecondary.empty()) return;

    std::vector<std::pair<BranchID, ThinnedAssociationBranches const*> > assocToBranches =
      fileAssociationsHelper.associationToBranches();

    for(BranchID const& association : associationsFromSecondary) {

      auto branches = std::lower_bound(assocToBranches.begin(), assocToBranches.end(),
                                       std::make_pair(association, static_cast<ThinnedAssociationBranches const*>(nullptr)),
                                        [](std::pair<BranchID, ThinnedAssociationBranches const*> const& x,
                                           std::pair<BranchID, ThinnedAssociationBranches const*> const& y)
                                          { return x.first < y.first; });
      // This should never happen
      if(branches == assocToBranches.end() || branches->first != association) {
        throw edm::Exception(errors::LogicError,
                             "ThinnedAssociationHelper::initAssociationsFromSecondary could not find branches information, contact Framework developers");
      }
      addAssociation(*(branches->second));
    }
    sort();
  }
}
