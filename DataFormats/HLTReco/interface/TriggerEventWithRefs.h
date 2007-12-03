#ifndef HLTReco_TriggerEventWithRefs_h
#define HLTReco_TriggerEventWithRefs_h

/** \class trigger::TriggerEventWithRefs
 *
 *  The single EDProduct to be saved for events (RAW case)
 *  describing the details of the (HLT) trigger table
 *
 *  $Date: 2007/12/03 14:21:03 $
 *  $Revision: 1.2 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include <string>
#include <vector>

namespace trigger
{

  /// The single EDProduct to be saved in addition for each event
  /// - but only in the RAW case
  class TriggerEventWithRefs {

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
      size_type others_;
      /// constructor
      TriggerFilterObject() :
	filterLabel_(),
	photons_(0), electrons_(0), muons_(0), jets_(0), composites_(0), mets_(0), hts_(0), others_(0) { }
      TriggerFilterObject(const std::string& filterLabel,
        size_type np, size_type ne, size_type nm, size_type nj, size_type nc, size_type nM, size_type nH, size_type nO) :
	filterLabel_(filterLabel),
	photons_(np), electrons_(ne), muons_(nm), jets_(nj), composites_(nc), mets_(nM), hts_(nH), others_(nO) { }
    };

  /// data members
  private:
    /// the filters recorded here
    std::vector<TriggerFilterObject> filterObjects_;
    /// non-owning pointers into collections (linearised)
    std::vector<reco::RecoEcalCandidateRef> photons_;
    std::vector<reco::ElectronRef> electrons_;
    std::vector<reco::RecoChargedCandidateRef> muons_;
    std::vector<reco::CaloJetRef> jets_;
    std::vector<reco::CompositeCandidateRef> composites_;
    std::vector<reco::CaloMETRef> mets_;
    std::vector<reco::METRef> hts_;
    std::vector<TriggerRef> others_;

  /// methods
  public:
    /// constructors
    TriggerEventWithRefs(): filterObjects_(),
      photons_(), electrons_(), muons_(), jets_(), composites_(), mets_(), hts_(), others_() { }

    /// setters - to build EDProduct
    void addFilterObject(const std::string filterLabel, const TriggerFilterObjectWithRefs& tfowr) {
      photons_.insert(photons_.end(),tfowr.getPhotons().begin(),tfowr.getPhotons().end());
      electrons_.insert(electrons_.end(),tfowr.getElectrons().begin(),tfowr.getElectrons().end());
      muons_.insert(muons_.end(),tfowr.getMuons().begin(),tfowr.getMuons().end());
      jets_.insert(jets_.end(),tfowr.getJets().begin(),tfowr.getJets().end());
      composites_.insert(composites_.end(),tfowr.getComposites().begin(),tfowr.getComposites().end());
      mets_.insert(mets_.end(),tfowr.getMETs().begin(),tfowr.getMETs().end());
      hts_.insert(hts_.end(),tfowr.getHTs().begin(),tfowr.getHTs().end());
      others_.insert(others_.end(),tfowr.getOthers().begin(),tfowr.getOthers().end());
      filterObjects_.push_back(
        TriggerFilterObject(filterLabel, 
			    photons_.size(), electrons_.size(), 
			    muons_.size(), jets_.size(), composites_.size(),
			    mets_.size(), hts_.size(), others_.size()
			   )
	);
    }

    /// getters - for user access
    size_type numFilters() const {return filterObjects_.size();}

    const std::string& getFilterLabel(size_type index) const {return filterObjects_.at(index).filterLabel_;}

    size_type find(const std::string& filterLabel) const {
      const size_type n(numFilters());
      for (size_type i=0; i!=n; ++i) {
	if (filterLabel==filterObjects_[i].filterLabel_) {return i;}
      }
      return n;
    }

    size_type numPhotons(size_type index) const {
      return filterObjects_.at(index).photons_ - (index==0? 0 : filterObjects_.at(index-1).photons_);
    }
    size_type numPhotons(const std::string& filterLabel) const {
      return numPhotons(find(filterLabel));
    }

    size_type numElectrons(size_type index) const {
      return filterObjects_.at(index).electrons_ - (index==0? 0 : filterObjects_.at(index-1).electrons_);
    }
    size_type numElectrons(const std::string& filterLabel) const {
      return numElectrons(find(filterLabel));
    }

    size_type numMuons(size_type index) const {
      return filterObjects_.at(index).muons_ - (index==0? 0 : filterObjects_.at(index-1).muons_);
    }
    size_type numMuons(const std::string& filterLabel) const {
      return numMuons(find(filterLabel));
    }

    size_type numJets(size_type index) const {
      return filterObjects_.at(index).jets_ - (index==0? 0 : filterObjects_.at(index-1).jets_);
    }
    size_type numJets(const std::string& filterLabel) const {
      return numJets(find(filterLabel));
    }

    size_type numComposites(size_type index) const {
      return filterObjects_.at(index).composites_ - (index==0? 0 : filterObjects_.at(index-1).composites_);
    }
    size_type numComposites(const std::string& filterLabel) const {
      return numComposites(find(filterLabel));
    }

    size_type numMETs(size_type index) const {
      return filterObjects_.at(index).mets_ - (index==0? 0 : filterObjects_.at(index-1).mets_);
    }
    size_type numMETs(const std::string& filterLabel) const {
      return numMETs(find(filterLabel));
    }

    size_type numHTs(size_type index) const {
      return filterObjects_.at(index).hts_ - (index==0? 0 : filterObjects_.at(index-1).hts_);
    }
    size_type numHTs(const std::string& filterLabel) const {
      return numHTs(find(filterLabel));
    }

    size_type numOthers(size_type index) const {
      return filterObjects_.at(index).others_ - (index==0? 0 : filterObjects_.at(index-1).others_);
    }
    size_type numOthers(const std::string& filterLabel) const {
      return numOthers(find(filterLabel));
    }

    /// iterators

    std::vector<reco::RecoEcalCandidateRef>::const_iterator
      photons_begin(size_type index) const
    { return photons_.begin() + 
      (index==0? 0 : filterObjects_.at(index-1).photons_);
    }
    std::vector<reco::RecoEcalCandidateRef>::const_iterator
      photons_end(size_type index) const
    { return photons_.begin() + filterObjects_.at(index).photons_; }


    std::vector<reco::ElectronRef>::const_iterator
      electrons_begin(size_type index) const
    { return electrons_.begin() + 
      (index==0? 0 : filterObjects_[index-1].electrons_);
    }
    std::vector<reco::ElectronRef>::const_iterator
      electrons_end(size_type index) const
    { return electrons_.begin() + filterObjects_[index].electrons_; }


    std::vector<reco::RecoChargedCandidateRef>::const_iterator
      muons_begin(size_type index) const
    { return muons_.begin() + 
      (index==0? 0 : filterObjects_[index-1].muons_);
    }
    std::vector<reco::RecoChargedCandidateRef>::const_iterator
      muons_end(size_type index) const
    { return muons_.begin() + filterObjects_[index].muons_; }


    std::vector<reco::CaloJetRef>::const_iterator
      jets_begin(size_type index) const
    { return jets_.begin() + 
      (index==0? 0 : filterObjects_[index-1].jets_);
    }
    std::vector<reco::CaloJetRef>::const_iterator
      jets_end(size_type index) const
    { return jets_.begin() + filterObjects_[index].jets_; }


    std::vector<reco::CompositeCandidateRef>::const_iterator
      composites_begin(size_type index) const
    { return composites_.begin() + 
      (index==0? 0 : filterObjects_[index-1].composites_);
    }
    std::vector<reco::CompositeCandidateRef>::const_iterator
      composites_end(size_type index) const
    { return composites_.begin() + filterObjects_[index].composites_; }


    std::vector<reco::CaloMETRef>::const_iterator
      mets_begin(size_type index) const
    { return mets_.begin() + 
      (index==0? 0 : filterObjects_[index-1].mets_);
    }
    std::vector<reco::CaloMETRef>::const_iterator
      mets_end(size_type index) const
    { return mets_.begin() + filterObjects_[index].mets_; }


    std::vector<reco::METRef>::const_iterator
      hts_begin(size_type index) const
    { return hts_.begin() + 
      (index==0? 0 : filterObjects_[index-1].hts_);
    }
    std::vector<reco::METRef>::const_iterator
      hts_end(size_type index) const
    { return hts_.begin() + filterObjects_[index].hts_; }


    std::vector<TriggerRef>::const_iterator
      others_begin(size_type index) const
    { return others_.begin() + 
      (index==0? 0 : filterObjects_[index-1].others_);
    }
    std::vector<TriggerRef>::const_iterator
      others_end(size_type index) const
    { return others_.begin() + filterObjects_[index].others_; }

  };

}

#endif
