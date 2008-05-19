#ifndef HLTReco_TriggerEvent_h
#define HLTReco_TriggerEvent_h

/** \class trigger::TriggerEvent
 *
 *  The single EDProduct to be saved for each event (AOD case)
 *  describing the (HLT) trigger table
 *
 *  $Date: 2008/05/02 13:35:27 $
 *  $Revision: 1.11 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <string>
#include <vector>

namespace trigger
{

  /// The single EDProduct to be saved for each event (AOD case)
  class TriggerEvent {

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
      TriggerFilterObject(): filterTag_(), filterIds_(), filterKeys_() {
	filterTag_=edm::InputTag().encode();
      }
      TriggerFilterObject(const edm::InputTag& filterTag): filterTag_(filterTag.encode()), filterIds_(), filterKeys_() { }
      TriggerFilterObject(const edm::InputTag& filterTag, const Vids& filterIds, const Keys& filterKeys): filterTag_(filterTag.encode()), filterIds_(filterIds), filterKeys_(filterKeys) { }
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
    TriggerEvent(): usedProcessName_(), collectionTags_(), collectionKeys_(), triggerObjects_(), triggerFilters_() { }
    TriggerEvent(const std::string& usedProcessName, size_type nc, size_type no, size_type nf):
      usedProcessName_(usedProcessName), 
      collectionTags_(),
      collectionKeys_(),
      triggerObjects_(), 
      triggerFilters_() 
    {
      collectionTags_.reserve(nc); collectionKeys_.reserve(nc);
      triggerObjects_.reserve(no); triggerFilters_.reserve(nf); 
    }

    /// setters
    void addObjects(const TriggerObjectCollection& triggerObjects) {triggerObjects_.insert(triggerObjects_.end(), triggerObjects.begin(), triggerObjects.end());}
    void addCollections(const std::vector<edm::InputTag>& collectionTags, const Keys& collectionKeys) {
      assert(collectionTags.size()==collectionKeys.size());
      const size_type n(collectionTags.size());
      for (size_type i=0; i!=n; ++i) {
	collectionTags_.push_back(collectionTags[i].encode());
      }
      collectionKeys_.insert(collectionKeys_.end(), collectionKeys.begin(), collectionKeys.end());
    }

    void addFilter(const edm::InputTag& filterTag, const Vids& filterIds, const Keys& filterKeys) {triggerFilters_.push_back(TriggerFilterObject(filterTag, filterIds, filterKeys));}

    /// getters
    const std::string& usedProcessName() const {return usedProcessName_;}
    const std::vector<std::string>& collectionTags() const {return collectionTags_;}
    const Keys& collectionKeys() const {return collectionKeys_;}
    const TriggerObjectCollection& getObjects() const {return triggerObjects_;}

    const edm::InputTag collectionTag(size_type index) const {return edm::InputTag(collectionTags_.at(index));}
    size_type collectionKey(size_type index) const {return collectionKeys_.at(index);}
    const edm::InputTag filterTag(size_type index) const {return edm::InputTag(triggerFilters_.at(index).filterTag_);}
    const Vids& filterIds(size_type index) const {return triggerFilters_.at(index).filterIds_;}
    const Keys& filterKeys(size_type index) const {return triggerFilters_.at(index).filterKeys_;}

    /// find index of collection from collection tag
    size_type collectionIndex(const edm::InputTag& collectionTag) const {
      const std::string encodedCollectionTag(collectionTag.encode());
      const size_type n(collectionTags_.size());
      for (size_type i=0; i!=n; ++i) {
	if (encodedCollectionTag==collectionTags_[i]) {return i;}
      }
      return n;
    }
    /// find index of filter in data-member vector from filter tag
    size_type filterIndex(const edm::InputTag& filterTag) const {
      const std::string encodedFilterTag(filterTag.encode());
      const size_type n(triggerFilters_.size());
      for (size_type i=0; i!=n; ++i) {
	if (encodedFilterTag==triggerFilters_[i].filterTag_) {return i;}
      }
      return n;
    }

    /// other
    size_type sizeCollections() const {return collectionTags_.size();}
    size_type sizeObjects() const {return triggerObjects_.size();}
    size_type sizeFilters() const {return triggerFilters_.size();}

  };

}

#endif
