#ifndef DataFormats_Provenance_BranchMapper_h
#define DataFormats_Provenance_BranchMapper_h

/*----------------------------------------------------------------------
  
BranchMapper: The mapping from per event product ID's to BranchID's.

----------------------------------------------------------------------*/
#include <iosfwd>
#include <set>
#include <map>
#include "boost/shared_ptr.hpp"

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "FWCore/Utilities/interface/Algorithms.h"

/*
  BranchMapper

*/

namespace edm {
  template <typename T>
  class BranchMapper {
  public:
    BranchMapper();

    explicit BranchMapper(bool delayedRead);

    virtual ~BranchMapper() {}

    void write(std::ostream& os) const;

    BranchID productToBranch(ProductID const& pid) const;
    
    boost::shared_ptr<T> branchToEntryInfo(BranchID const& bid) const;

    void insert(T const& entryInfo);

    void mergeMappers(boost::shared_ptr<BranchMapper<T> > other) {nextMapper_ = other;}


  private:
    void readProvenance() const;
    virtual void readProvenance_() const {}

    typedef typename std::set<T> eiSet;
    typedef typename std::map<ProductID, typename eiSet::const_iterator> eiMap;

    eiSet entryInfoSet_;

    eiMap entryInfoMap_;

    boost::shared_ptr<BranchMapper<T> > nextMapper_;

    mutable bool delayedRead_;

  };
  
  template <typename T>
  inline
  std::ostream&
  operator<<(std::ostream& os, BranchMapper<T> const& p) {
    p.write(os);
    return os;
  }

  template <typename T>
  inline
  BranchMapper<T>::BranchMapper() :
    entryInfoSet_(),
    entryInfoMap_(),
    nextMapper_(),
    delayedRead_(false)
  { }

  template <typename T>
  inline
  BranchMapper<T>::BranchMapper(bool delayedRead) :
    entryInfoSet_(),
    entryInfoMap_(),
    nextMapper_(),
    delayedRead_(delayedRead)
  { }

  template <typename T>
  inline
  void
  BranchMapper<T>::readProvenance() const {
    if (delayedRead_) {
      delayedRead_ = false;
      readProvenance_();
    }
  }

  template <typename T>
  inline
  void
  BranchMapper<T>::insert(T const& entryInfo) {
    readProvenance();
    entryInfoSet_.insert(entryInfo);
    if (!entryInfoMap_.empty()) {
      entryInfoMap_.insert(std::make_pair(entryInfo.productID(), entryInfoSet_.find(entryInfo)));
    }
  }
    
  template <typename T>
  inline
  boost::shared_ptr<T>
  BranchMapper<T>::branchToEntryInfo(BranchID const& bid) const {
    readProvenance();
    T ei(bid);
    typename eiSet::const_iterator it = entryInfoSet_.find(ei);
    if (it == entryInfoSet_.end()) {
      if (nextMapper_) {
	return nextMapper_->branchToEntryInfo(bid);
      } else {
	return boost::shared_ptr<T>();
      }
    }
    return boost::shared_ptr<T>(new T(*it));
  }

/*
  template <typename T>
  inline
  ProductID 
  BranchMapper<T>::branchToProduct(BranchID const& bid) const {
    readProvenance();
    T ei(bid);
    typename eiSet::const_iterator it = entryInfoSet_.find(ei);
    if (it == entryInfoSet_.end()) {
      if (nextMapper_) {
	return nextMapper_->branchToProduct(bid);
      } else {
	return ProductID();
      }
    }
    return it->productID();
  }
*/

  template <typename T>
  inline
  BranchID 
  BranchMapper<T>::productToBranch(ProductID const& pid) const {
    readProvenance();
    if (entryInfoMap_.empty()) {
      eiMap & map = const_cast<eiMap &>(entryInfoMap_);
      for (typename eiSet::const_iterator i = entryInfoSet_.begin(), iEnd = entryInfoSet_.end();
	  i != iEnd; ++i) {
	map.insert(std::make_pair(i->productID(), i));
      }
    }
    typename eiMap::const_iterator it = entryInfoMap_.find(pid);
    if (it == entryInfoMap_.end()) {
      if (nextMapper_) {
	return nextMapper_->productToBranch(pid);
      } else {
	return BranchID();
      }
    }
    return it->second->branchID();
  }

}
#endif
