#ifndef HLTReco_TriggerEventWithRefs_h
#define HLTReco_TriggerEventWithRefs_h

/** \class trigger::TriggerEventWithRefs
 *
 *  The single EDProduct to be saved for events (RAW case)
 *  describing the details of the (HLT) trigger table
 *
 *  $Date: 2007/12/05 14:24:02 $
 *  $Revision: 1.8 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include <string>
#include <vector>

namespace trigger
{

  /// The single EDProduct to be saved in addition for each event
  /// - but only in the "RAW" case: for a fraction of all events

  class TriggerEventWithRefs : public TriggerRefsCollections {

  private:

    /// Helper class: trigger objects firing a single filter
    class TriggerFilterObject {
    public:
      /// label of filter
      std::string filterLabel_;
      /// 1-after-end (std C++) indices into linearised vector of Refs
      /// (-> first start index is always 0)
      size_type photons_;
      size_type electrons_;
      size_type muons_;
      size_type jets_;
      size_type composites_;
      size_type mets_;
      size_type hts_;

      /// constructor
      TriggerFilterObject() :
	filterLabel_(),
	photons_(0), electrons_(0), muons_(0), jets_(0), composites_(0), mets_(0), hts_(0) { }
      TriggerFilterObject(const std::string& filterLabel,
        size_type np, size_type ne, size_type nm, size_type nj, size_type nc, size_type nM, size_type nH) :
	filterLabel_(filterLabel),
	photons_(np), electrons_(ne), muons_(nm), jets_(nj), composites_(nc), mets_(nM), hts_(nH) { }
    };

  /// data members
  private:
    /// the filters recorded here
    std::vector<TriggerFilterObject> filterObjects_;

  /// methods
  public:
    /// constructors
    TriggerEventWithRefs(): TriggerRefsCollections(), filterObjects_() { }

    /// setters - to build EDProduct
    void addFilterObject(const std::string& filterLabel, const TriggerFilterObjectWithRefs& tfowr) {
      filterObjects_.push_back(
        TriggerFilterObject(filterLabel, 
			    append(tfowr.photonIds(),tfowr.photonRefs()),
			    append(tfowr.electronIds(),tfowr.electronRefs()),
			    append(tfowr.muonIds(),tfowr.muonRefs()),
			    append(tfowr.jetIds(),tfowr.jetRefs()),
			    append(tfowr.compositeIds(),tfowr.compositeRefs()),
			    append(tfowr.metIds(),tfowr.metRefs()),
			    append(tfowr.htIds(),tfowr.htRefs())
			   )
	);
    }

    /// getters - for user access

    /// number of filters
    size_type size() const {return filterObjects_.size();}

    /// label from index
    const std::string& filterLabel(size_type filterIndex) const {
      return filterObjects_.at(filterIndex).filterLabel_;
    }

    /// index from label
    size_type filterIndex(const std:: string filterLabel) const {
      const size_type n(filterObjects_.size());
      for (size_type i=0; i!=n; ++i) {
	if (filterLabel==filterObjects_[i].filterLabel_) {return i;}
      }
      return n;
    }

    /// extract Ref<C>s for a specific filter and of specific physics type

    void getObjects(size_type filter, int id, VRphoton& photons) const {
      const size_type begin(filter==0? 0 : filterObjects_.at(filter-1).photons_);
      const size_type end(filterObjects_.at(filter).photons_);
      TriggerRefsCollections::getObjects(id,photons,begin,end);
    }

    void getObjects(size_type filter, int id, VRelectron& electrons) const {
      const size_type begin(filter==0? 0 : filterObjects_.at(filter-1).electrons_);
      const size_type end(filterObjects_.at(filter).electrons_);
      TriggerRefsCollections::getObjects(id,electrons,begin,end);
    }

    void getObjects(size_type filter, int id, VRmuon& muons) const {
      const size_type begin(filter==0? 0 : filterObjects_.at(filter-1).muons_);
      const size_type end(filterObjects_.at(filter).muons_);
      TriggerRefsCollections::getObjects(id,muons,begin,end);
    }

    void getObjects(size_type filter, int id, VRjet& jets) const {
      const size_type begin(filter==0? 0 : filterObjects_.at(filter-1).jets_);
      const size_type end(filterObjects_.at(filter).jets_);
      TriggerRefsCollections::getObjects(id,jets,begin,end);
    }

    void getObjects(size_type filter, int id, VRcomposite& composites) const {
      const size_type begin(filter==0? 0 : filterObjects_.at(filter-1).composites_);
      const size_type end(filterObjects_.at(filter).composites_);
      TriggerRefsCollections::getObjects(id,composites,begin,end);
    }

    void getObjects(size_type filter, int id, VRmet& mets) const {
      const size_type begin(filter==0? 0 : filterObjects_.at(filter-1).mets_);
      const size_type end(filterObjects_.at(filter).mets_);
      TriggerRefsCollections::getObjects(id,mets,begin,end);
    }

    void getObjects(size_type filter, int id, VRht& hts) const {
      const size_type begin(filter==0? 0 : filterObjects_.at(filter-1).hts_);
      const size_type end(filterObjects_.at(filter).hts_);
      TriggerRefsCollections::getObjects(id,hts,begin,end);
    }

  };

}

#endif
