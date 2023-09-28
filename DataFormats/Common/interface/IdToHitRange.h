#ifndef DataFormats_Common_IdToHitRange_h
#define DataFormats_Common_IdToHitRange_h
// -*- C++ -*-
//
// Package:     DataFormats/Common
// Class  :     IdToHitRange
//
/**\class IdToHitRange IdToHitRange.h "DataFormats/Common/interface/IdToHitRange.h"

 Description: Used to associate an ID to a range of hits

 Usage:
    Allows quick lookup of a range of hits via the ID

*/
//
// Original Author:  Christopher Jones
//         Created:  Tue, 26 Sep 2023 20:41:14 GMT
//

// system include files
#include <vector>
#include <algorithm>

// user include files
#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "FWCore/Utilities/interface/EDMException.h"

// forward declarations

namespace edm {
  template <typename ID, typename HIT>
  class IdToHitRange {
  public:
    using container = std::vector<HIT>;
    /// contained object type
    using value_type = typename container::value_type;
    /// collection size type
    using size_type = typename container::size_type;
    /// reference type
    using reference = typename container::reference;
    /// pointer type
    using pointer = typename container::pointer;
    /// constant access iterator type
    using const_iterator = typename container::const_iterator;
    /// iterator range
    using range = std::pair<const_iterator, const_iterator>;

    using id_iterator = typename std::vector<ID>::const_iterator;

  public:
    IdToHitRange() {}

    // ---------- const member functions ---------------------
    /// get a range of objects with specified identifier
    range get(ID id) const {
      auto i = find(id);
      if (i == ids_.end()) {
        return {collection_.end(), collection_.end()};
      }
      auto offsetInOffsets = i - ids_.begin();
      if (ids_.size() != offsets_.size()) {
        offsetInOffsets = offsets_[offsetInOffsets] + ids_.size();
      }
      auto startIndex = offsets_[offsetInOffsets];
      if (static_cast<std::size_t>(offsetInOffsets + 1) == offsets_.size()) {
        return {collection_.begin() + startIndex, collection_.end()};
      }
      auto endIndex = offsets_[offsetInOffsets + 1];
      return {collection_.begin() + startIndex, collection_.begin() + endIndex};
    }

    /// get range of objects matching a specified identifier with a specified comparator.
    /// <b>WARNING</b>: the comparator has to be written
    /// in such a way that the std::equal_range
    /// function returns a meaningful range.
    /// Not properly written comparators may return
    /// an unpredictable range. It is recommended
    /// to use only comparators provided with CMSSW release.
    template <typename CMP>
    range get(ID id, CMP comparator) const {
      if (ids_.size() != offsets_.size()) {
        throw edm::Exception(edm::errors::LogicError, "calling get with comparitor before sorting.");
      }
      auto r = std::equal_range(ids_.begin(), ids_.end(), id, comparator);
      const_iterator begin, end;
      if ((r.first) == ids_.end()) {
        begin = end = collection_.end();
        return std::make_pair(begin, end);
      } else {
        begin = collection_.begin() + offsets_[r.first - ids_.begin()];
      }
      if ((r.second) == ids_.end()) {
        end = collection_.end();
      } else {
        end = collection_.begin() + offsets_[r.second - ids_.begin()];
      }
      return std::make_pair(begin, end);
    }
    /// get range of objects matching a specified identifier with a specified comparator.
    template <typename CMP>
    range get(std::pair<ID, CMP> p) const {
      return get(p.first, p.second);
    }

    /// return number of contained object
    size_t size() const { return collection_.size(); }
    /// first collection iterator
    /// hits in collection are not guaranteed to be in ID order
    const_iterator begin() const { return collection_.begin(); }
    /// last collection iterator
    const_iterator end() const { return collection_.end(); }

    // ---------- member functions ---------------------------
    /// insert an object range with specified identifier
    template <typename CI>
    void put(ID id, CI begin, CI end) {
      auto i = std::lower_bound(ids_.begin(), ids_.end(), id);
      if (i == ids_.end()) {
        bool isAlreadySorted = (offsets_.size() == ids_.size());
        ids_.emplace_back(std::move(id));
        offsets_.push_back(collection_.size());
        collection_.insert(collection_.end(), begin, end);
        if (not isAlreadySorted) {
          //need to update the indirection table used to find the ranges
          offsets_.insert(offsets_.begin() + ids_.size() - 1, offsets_.size() - ids_.size());
        }
        return;
      } else if (*i == id) {
        throw edm::Exception(edm::errors::LogicError, "trying to insert duplicate entry");
      }
      if (offsets_.size() == ids_.size()) {
        //was sorted, but now it will not be so need to reformat offsets_
        std::vector<unsigned int> newOffsets;
        newOffsets.reserve(ids_.size() * 2 + 2);
        for (std::size_t index = 0; index < ids_.size(); ++index) {
          newOffsets.push_back(index);
        }
        for (std::size_t index = 0; index < ids_.size(); ++index) {
          newOffsets.push_back(offsets_[index]);
        }
        offsets_ = std::move(newOffsets);
      }
      auto offsetInIds = i - ids_.begin();
      ids_.insert(i, id);
      offsets_.push_back(collection_.size());
      collection_.insert(collection_.end(), begin, end);
      offsets_.insert(offsets_.begin() + offsetInIds, offsets_.size() - ids_.size());
    }

    /// perfor post insert action
    void post_insert() {
      if (ids_.size() == offsets_.size()) {
        //already sorted
        return;
      }

      std::vector<HIT> sortedCollection;
      sortedCollection.reserve(collection_.size());

      std::vector<unsigned int> offsets;
      offsets.reserve(ids_.size());

      for (auto id : ids_) {
        offsets.push_back(sortedCollection.size());
        auto range = get(id);
        sortedCollection.insert(sortedCollection.end(), range.first, range.second);
      }
      offsets_ = std::move(offsets);
      collection_ = std::move(sortedCollection);
    }

    /// first identifier iterator
    id_iterator id_begin() const { return ids_.begin(); }
    /// last identifier iterator
    id_iterator id_end() const { return ids_.end(); }
    /// number of contained identifiers
    size_t id_size() const { return ids_.size(); }
    /// indentifier vector
    std::vector<ID> ids() const { return ids_; }
    /// direct access to an object in the collection
    reference operator[](size_type i) { return collection_[i]; }

    /// swap member function
    void swap(IdToHitRange<ID, HIT>& other);

    //Used by ROOT storage
    CMS_CLASS_VERSION(3)

  private:
    typename std::vector<ID>::const_iterator find(ID id) const {
      auto i = std::lower_bound(ids_.begin(), ids_.end(), id);
      if (i == ids_.end() or *i != id) {
        return ids_.end();
      }
      return i;
    }

    // ---------- member data --------------------------------
    std::vector<ID> ids_;
    std::vector<unsigned int> offsets_;
    container collection_;
  };

  template <typename ID, typename HIT>
  inline void IdToHitRange<ID, HIT>::swap(IdToHitRange<ID, HIT>& other) {
    ids_.swap(other.ids_);
    offsets_.swap(other.offsets_);
    collection_.swap(other.collection_);
  }

  // free swap function
  template <typename ID, typename HIT>
  inline void swap(IdToHitRange<ID, HIT>& a, IdToHitRange<ID, HIT>& b) {
    a.swap(b);
  }

}  // namespace edm

#endif
