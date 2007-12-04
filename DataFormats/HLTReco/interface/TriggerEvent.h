#ifndef HLTReco_TriggerEvent_h
#define HLTReco_TriggerEvent_h

/** \class trigger::TriggerEvent
 *
 *  The single EDProduct to be saved for each event (AOD case)
 *  describing the (HLT) trigger table
 *
 *  $Date: 2007/12/04 08:35:53 $
 *  $Revision: 1.2 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include <string>
#include <vector>

namespace trigger
{

  /// The single EDProduct to be saved for each event (AOD case)
  class TriggerEvent {

  private:

    /// Helper class: recording trigger objects firing a single filter
    class TriggerFilterObject {
    public:
      /// the label of the filter
      std::string filterLabel_;
      /// indices pointing into collection of unique TriggerObjects
      Keys filterKeys_;
      /// constructors
      TriggerFilterObject(): filterLabel_(), filterKeys_() { }
      TriggerFilterObject(const std::string& filterLabel): filterLabel_(filterLabel), filterKeys_() { }
      TriggerFilterObject(const std::string& filterLabel, const Keys& filterKeys): filterLabel_(filterLabel), filterKeys_(filterKeys) { }
    };

  /// data members
  private:
    /// collection of all unique physics objects (linearised vector)
    TriggerObjectCollection triggerObjects_;
    /// collection of all TriggerFilterObjects
    std::vector<TriggerFilterObject> triggerFilters_;

  ///methods
  public:
    /// constructors
    TriggerEvent(): triggerObjects_(), triggerFilters_() { }

    /// setters
    void addObjects(const TriggerObjectCollection& triggerObjects) {triggerObjects_.insert(triggerObjects_.end(), triggerObjects.begin(), triggerObjects.end());}
    void addFilter(const std::string& filterLabel, const Keys& filterKeys) {triggerFilters_.push_back(TriggerFilterObject(filterLabel, filterKeys));}

    /// getters
    const TriggerObjectCollection& getObjects() const {return triggerObjects_;}
    const std::string& getFilterLabel(size_type index) const {return triggerFilters_.at(index).filterLabel_;}
    const Keys& getFilterKeys(size_type index) const {return triggerFilters_.at(index).filterKeys_;}

    /// find index of filter in data-member vector from filter label
    size_type find(const std::string& filterLabel) const {
      const size_type n(triggerFilters_.size());
      for (size_type i=0; i!=n; ++i) {
	if (filterLabel==triggerFilters_[i].filterLabel_) {return i;}
      }
      return n;
    }

    /// other
    size_type numObjects() const {return triggerObjects_.size();}
    size_type numFilters() const {return triggerFilters_.size();}

  };

}

#endif
