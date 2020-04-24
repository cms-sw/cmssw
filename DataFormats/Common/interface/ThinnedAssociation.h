#ifndef DataFormats_Common_ThinnedAssociation_h
#define DataFormats_Common_ThinnedAssociation_h

/** \class edm::ThinnedAssociation
\author W. David Dagenhart, created 11 June 2014
*/

#include "DataFormats/Provenance/interface/ProductID.h"

#include <vector>

namespace edm {

  class ThinnedAssociation {
  public:

    ThinnedAssociation();

    ProductID const& parentCollectionID() const { return parentCollectionID_; }
    ProductID const& thinnedCollectionID() const { return thinnedCollectionID_; }
    std::vector<unsigned int> const& indexesIntoParent() const { return indexesIntoParent_; }

    bool hasParentIndex(unsigned int parentIndex, unsigned int& thinnedIndex) const;

    void setParentCollectionID(ProductID const& v) { parentCollectionID_ = v; }
    void setThinnedCollectionID(ProductID const& v) { thinnedCollectionID_ = v; }
    void push_back(unsigned int index) { indexesIntoParent_.push_back(index); }

  private:

    ProductID parentCollectionID_;
    ProductID thinnedCollectionID_;

    // The size of indexesIntoParent_ is the same as
    // the size of the thinned collection and each
    // element of indexesIntoParent corresponds to the
    // element of the thinned collection at the same position.
    // The values give the index of the corresponding element
    // in the parent collection.
    std::vector<unsigned int> indexesIntoParent_;
  };
}
#endif
