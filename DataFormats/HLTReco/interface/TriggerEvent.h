#ifndef HLTReco_TriggerEvent_h
#define HLTReco_TriggerEvent_h

/** \class trigger::TriggerEvent
 *
 *  The single EDProduct to be saved for each event (AOD case)
 *  describing the (HLT) trigger table
 *
 *  $Date: 2008/05/02 12:08:41 $
 *  $Revision: 1.10 $
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
    /// collection of all unique physics objects (linearised vector)
    TriggerObjectCollection triggerObjects_;
    /// collection of all TriggerFilterObjects
    std::vector<TriggerFilterObject> triggerFilters_;

  ///methods
  public:
    /// constructors
    TriggerEvent(): usedProcessName_(), triggerObjects_(), triggerFilters_() { }
    TriggerEvent(const std::string& usedProcessName, size_type no, size_type nf):
      usedProcessName_(usedProcessName), 
      triggerObjects_(), 
      triggerFilters_() 
    {
      triggerObjects_.reserve(no); triggerFilters_.reserve(nf); 
    }

    /// setters
    void addObjects(const TriggerObjectCollection& triggerObjects) {triggerObjects_.insert(triggerObjects_.end(), triggerObjects.begin(), triggerObjects.end());}
    void addFilter(const edm::InputTag& filterTag, const Vids& filterIds, const Keys& filterKeys) {triggerFilters_.push_back(TriggerFilterObject(filterTag, filterIds, filterKeys));}

    /// getters
    const std::string& usedProcessName() const {return usedProcessName_;}
    const TriggerObjectCollection& getObjects() const {return triggerObjects_;}
    const edm::InputTag filterTag(size_type index) const {return edm::InputTag(triggerFilters_.at(index).filterTag_);}
    const Vids& filterIds(size_type index) const {return triggerFilters_.at(index).filterIds_;}
    const Keys& filterKeys(size_type index) const {return triggerFilters_.at(index).filterKeys_;}

    /// find index of filter in data-member vector from filter label
    size_type filterIndex(const edm::InputTag& filterTag) const {
      const std::string encodedFilterTag(filterTag.encode());
      const size_type n(triggerFilters_.size());
      for (size_type i=0; i!=n; ++i) {
	if (encodedFilterTag==triggerFilters_[i].filterTag_) {return i;}
      }
      return n;
    }

    /// other
    size_type sizeObjects() const {return triggerObjects_.size();}
    size_type sizeFilters() const {return triggerFilters_.size();}

  };

}

#endif
