#ifndef HLTReco_TriggerEventWithRefs_h
#define HLTReco_TriggerEventWithRefs_h

/** \class trigger::TriggerEventWithRefs
 *
 *  The single EDProduct to be saved for events (RAW case)
 *  describing the details of the (HLT) trigger table
 *
 *  $Date: 2008/05/02 12:08:41 $
 *  $Revision: 1.16 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <string>
#include <vector>

namespace trigger
{

  /// The single EDProduct to be saved in addition for each event
  /// - but only in the "RAW" case: for a fraction of all events

  class TriggerEventWithRefs : public TriggerRefsCollections {

  public:

    /// Helper class: trigger objects firing a single filter
    class TriggerFilterObject {
    public:
      /// encoded InputTag of filter product
      std::string filterTag_;
      /// 1-after-end (std C++) indices into linearised vector of Refs
      /// (-> first start index is always 0)
      size_type photons_;
      size_type electrons_;
      size_type muons_;
      size_type jets_;
      size_type composites_;
      size_type mets_;
      size_type hts_;
      size_type pixtracks_;
      size_type l1em_;
      size_type l1muon_;
      size_type l1jet_;
      size_type l1etmiss_;

      /// constructor
      TriggerFilterObject() :
	filterTag_(),
	photons_(0), electrons_(0), muons_(0), jets_(0), composites_(0), mets_(0), hts_(0), pixtracks_(0), l1em_(0), l1muon_(0), l1jet_(0), l1etmiss_(0) {
      filterTag_=edm::InputTag().encode();
      }
      TriggerFilterObject(const edm::InputTag& filterTag,
        size_type np, size_type ne, size_type nm, size_type nj, size_type nc, size_type nM, size_type nH, size_type nt, size_type l1em, size_type l1muon, size_type l1jet, size_type l1etmiss) :
	filterTag_(filterTag.encode()),
	photons_(np), electrons_(ne), muons_(nm), jets_(nj), composites_(nc), mets_(nM), hts_(nH), pixtracks_(nt), l1em_(l1em), l1muon_(l1muon), l1jet_(l1jet), l1etmiss_(l1etmiss) { }
    };

  /// data members
  private:
    /// processName used to select products packed up
    std::string usedProcessName_;    
    /// the filters recorded here
    std::vector<TriggerFilterObject> filterObjects_;

  /// methods
  public:
    /// constructors
    TriggerEventWithRefs(): TriggerRefsCollections(), usedProcessName_(), filterObjects_() { }
    TriggerEventWithRefs(const std::string& usedProcessName, size_type n):
      TriggerRefsCollections(),
      usedProcessName_(usedProcessName),
      filterObjects_()
    {
      filterObjects_.reserve(n);
    }

    /// setters - to build EDProduct
    void addFilterObject(const edm::InputTag& filterTag, const TriggerFilterObjectWithRefs& tfowr) {
      filterObjects_.push_back(
        TriggerFilterObject(filterTag, 
			    addObjects(tfowr.photonIds(),tfowr.photonRefs()),
			    addObjects(tfowr.electronIds(),tfowr.electronRefs()),
			    addObjects(tfowr.muonIds(),tfowr.muonRefs()),
			    addObjects(tfowr.jetIds(),tfowr.jetRefs()),
			    addObjects(tfowr.compositeIds(),tfowr.compositeRefs()),
			    addObjects(tfowr.metIds(),tfowr.metRefs()),
			    addObjects(tfowr.htIds(),tfowr.htRefs()),
			    addObjects(tfowr.pixtrackIds(),tfowr.pixtrackRefs()),
			    addObjects(tfowr.l1emIds(),tfowr.l1emRefs()),
			    addObjects(tfowr.l1muonIds(),tfowr.l1muonRefs()),
			    addObjects(tfowr.l1jetIds(),tfowr.l1jetRefs()),
			    addObjects(tfowr.l1etmissIds(),tfowr.l1etmissRefs())
			   )
	);
    }

    /// getters - for user access
    const std::string& usedProcessName() const {return usedProcessName_;}

    /// number of filters
    size_type size() const {return filterObjects_.size();}

    /// tag from index
    const edm::InputTag filterTag(size_type filterIndex) const {
      return edm::InputTag(filterObjects_.at(filterIndex).filterTag_);
    }

    /// index from tag
    size_type filterIndex(const edm::InputTag& filterTag) const {
      const std::string encodedFilterTag (filterTag.encode());
      const size_type n(filterObjects_.size());
      for (size_type i=0; i!=n; ++i) {
	if (encodedFilterTag==filterObjects_[i].filterTag_) {return i;}
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

    void getObjects(size_type filter, int id, VRpixtrack& pixtracks) const {
      const size_type begin(filter==0? 0 : filterObjects_.at(filter-1).pixtracks_);
      const size_type end(filterObjects_.at(filter).pixtracks_);
      TriggerRefsCollections::getObjects(id,pixtracks,begin,end);
    }

    void getObjects(size_type filter, int id, VRl1em& l1em) const {
      const size_type begin(filter==0? 0 : filterObjects_.at(filter-1).l1em_);
      const size_type end(filterObjects_.at(filter).l1em_);
      TriggerRefsCollections::getObjects(id,l1em,begin,end);
    }
    void getObjects(size_type filter, int id, VRl1muon& l1muon) const {
      const size_type begin(filter==0? 0 : filterObjects_.at(filter-1).l1muon_);
      const size_type end(filterObjects_.at(filter).l1muon_);
      TriggerRefsCollections::getObjects(id,l1muon,begin,end);
    }
    void getObjects(size_type filter, int id, VRl1jet& l1jet) const {
      const size_type begin(filter==0? 0 : filterObjects_.at(filter-1).l1jet_);
      const size_type end(filterObjects_.at(filter).l1jet_);
      TriggerRefsCollections::getObjects(id,l1jet,begin,end);
    }
    void getObjects(size_type filter, int id, VRl1etmiss& l1etmiss) const {
      const size_type begin(filter==0? 0 : filterObjects_.at(filter-1).l1etmiss_);
      const size_type end(filterObjects_.at(filter).l1etmiss_);
      TriggerRefsCollections::getObjects(id,l1etmiss,begin,end);
    }

  };

}

#endif
