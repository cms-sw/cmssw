#ifndef HLTReco_TriggerEvent_h
#define HLTReco_TriggerEvent_h

/** \class trigger::TriggerEvent
 *
 *  The single EDProduct to be saved for each event (AOD case)
 *  describing the (HLT) trigger table
 *
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/traits.h"
#include <string>
#include <vector>
#include <cassert>

namespace trigger {

  /// The single EDProduct to be saved for each event (AOD case)
  class TriggerEvent : public edm::DoNotRecordParents {
  public:
    /// Helper class: recording trigger objects firing a single filter
    class TriggerFilterObject {
    public:
      /// encoded InputTag of filter product
      std::string filterTag_;
      /// physics object type as per filter
      Vids filterIds_;
      /// indices pointing into collection of unique TriggerObjects
      Keys filterKeys_;
      /// constructors
      TriggerFilterObject() : filterTag_(), filterIds_(), filterKeys_() { filterTag_ = edm::InputTag().encode(); }
      TriggerFilterObject(const edm::InputTag& filterTag)
          : filterTag_(filterTag.encode()), filterIds_(), filterKeys_() {}
      TriggerFilterObject(const edm::InputTag& filterTag, const Vids& filterIds, const Keys& filterKeys)
          : filterTag_(filterTag.encode()), filterIds_(filterIds), filterKeys_(filterKeys) {}
    };

    /// data members
  private:
    /// processName used to select products packed up
    std::string usedProcessName_;
    /// Input tags of packed up collections
    std::vector<std::string> collectionTags_;
    /// 1-past-end indices into linearised vector
    Keys collectionKeys_;
    /// collection of all unique physics objects (linearised vector)
    TriggerObjectCollection triggerObjects_;
    /// collection of all TriggerFilterObjects
    std::vector<TriggerFilterObject> triggerFilters_;

    ///methods
  public:
    /// constructors
    TriggerEvent() : usedProcessName_(), collectionTags_(), collectionKeys_(), triggerObjects_(), triggerFilters_() {}
    TriggerEvent(const std::string& usedProcessName, trigger::size_type nc, trigger::size_type no, trigger::size_type nf)
        : usedProcessName_(usedProcessName),
          collectionTags_(),
          collectionKeys_(),
          triggerObjects_(),
          triggerFilters_() {
      collectionTags_.reserve(nc);
      collectionKeys_.reserve(nc);
      triggerObjects_.reserve(no);
      triggerFilters_.reserve(nf);
    }

    /// setters
    void addObjects(const TriggerObjectCollection& triggerObjects) {
      triggerObjects_.insert(triggerObjects_.end(), triggerObjects.begin(), triggerObjects.end());
    }

    void addCollections(const std::vector<edm::InputTag>& collectionTags, const Keys& collectionKeys) {
      assert(collectionTags.size() == collectionKeys.size());
      const trigger::size_type n(collectionTags.size());
      for (trigger::size_type i = 0; i != n; ++i) {
        collectionTags_.push_back(collectionTags[i].encode());
      }
      collectionKeys_.insert(collectionKeys_.end(), collectionKeys.begin(), collectionKeys.end());
    }

    void addCollections(const std::vector<std::string>& collectionTags, const Keys& collectionKeys) {
      assert(collectionTags.size() == collectionKeys.size());
      collectionTags_.insert(collectionTags_.end(), collectionTags.begin(), collectionTags.end());
      collectionKeys_.insert(collectionKeys_.end(), collectionKeys.begin(), collectionKeys.end());
    }

    void addFilter(const edm::InputTag& filterTag, const Vids& filterIds, const Keys& filterKeys) {
      triggerFilters_.push_back(TriggerFilterObject(filterTag, filterIds, filterKeys));
    }

    /// getters
    const std::string& usedProcessName() const { return usedProcessName_; }
    const std::vector<std::string>& collectionTags() const { return collectionTags_; }
    const Keys& collectionKeys() const { return collectionKeys_; }
    const TriggerObjectCollection& getObjects() const { return triggerObjects_; }

    const edm::InputTag collectionTag(trigger::size_type index) const {
      return edm::InputTag(collectionTags_.at(index));
    }
    const std::string& collectionTagEncoded(trigger::size_type index) const { return collectionTags_.at(index); }
    trigger::size_type collectionKey(trigger::size_type index) const { return collectionKeys_.at(index); }
    const edm::InputTag filterTag(trigger::size_type index) const {
      return edm::InputTag(triggerFilters_.at(index).filterTag_);
    }
    const std::string& filterTagEncoded(trigger::size_type index) const { return triggerFilters_.at(index).filterTag_; }
    std::string filterLabel(trigger::size_type index) const {
      const std::string& tag = triggerFilters_.at(index).filterTag_;
      std::string::size_type idx = tag.find(':');
      return (idx == std::string::npos ? tag : tag.substr(0, idx));
    }
    const Vids& filterIds(trigger::size_type index) const { return triggerFilters_.at(index).filterIds_; }
    const Keys& filterKeys(trigger::size_type index) const { return triggerFilters_.at(index).filterKeys_; }

    /// find index of collection from collection tag
    trigger::size_type collectionIndex(const edm::InputTag& collectionTag) const {
      const std::string encodedCollectionTag(collectionTag.encode());
      const trigger::size_type n(collectionTags_.size());
      for (trigger::size_type i = 0; i != n; ++i) {
        if (encodedCollectionTag == collectionTags_[i]) {
          return i;
        }
      }
      return n;
    }
    /// find index of filter in data-member vector from filter tag
    trigger::size_type filterIndex(const edm::InputTag& filterTag) const {
      const std::string encodedFilterTag(filterTag.encode());
      const trigger::size_type n(triggerFilters_.size());
      for (trigger::size_type i = 0; i != n; ++i) {
        if (encodedFilterTag == triggerFilters_[i].filterTag_) {
          return i;
        }
      }
      return n;
    }

    /// other
    trigger::size_type sizeCollections() const { return collectionTags_.size(); }
    trigger::size_type sizeObjects() const { return triggerObjects_.size(); }
    trigger::size_type sizeFilters() const { return triggerFilters_.size(); }
  };

}  // namespace trigger

#endif
