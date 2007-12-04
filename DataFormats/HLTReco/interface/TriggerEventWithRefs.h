#ifndef HLTReco_TriggerEventWithRefs_h
#define HLTReco_TriggerEventWithRefs_h

/** \class trigger::TriggerEventWithRefs
 *
 *  The single EDProduct to be saved for events (RAW case)
 *  describing the details of the (HLT) trigger table
 *
 *  $Date: 2007/12/04 08:35:53 $
 *  $Revision: 1.5 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Provenance/interface/ProductID.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Candidate/interface/CompositeCandidateFwd.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/METReco/interface/METFwd.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include <string>
#include <vector>

namespace trigger
{

  /// The single EDProduct to be saved in addition for each event
  /// - but only in the "RAW" case: for a fraction of all events
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
    std::vector<XRef> others_;

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

    /// number of filters stored
    size_type numFilters() const {return filterObjects_.size();}

    /// label of ith filter
    const std::string& getFilterLabel(size_type index) const {
      return filterObjects_.at(index).filterLabel_;
    }

    /// find index of filter in data-member vector from filter label
    size_type find(const std::string& filterLabel) const {
      const size_type n(filterObjects_.size());
      for (size_type i=0; i!=n; ++i) {
	if (filterLabel==filterObjects_[i].filterLabel_) {return i;}
      }
      return n;
    }

    /// number of photons for a specific filter
    size_type numPhotons(size_type index) const {
      return filterObjects_.at(index).photons_ - (index==0? 0 : filterObjects_.at(index-1).photons_);
    }

    /// number of electrons for a specific filter
    size_type numElectrons(size_type index) const {
      return filterObjects_.at(index).electrons_ - (index==0? 0 : filterObjects_.at(index-1).electrons_);
    }

    /// number of muonsfor a specific filter
    size_type numMuons(size_type index) const {
      return filterObjects_.at(index).muons_ - (index==0? 0 : filterObjects_.at(index-1).muons_);
    }

    /// number of jets for a specific filter
    size_type numJets(size_type index) const {
      return filterObjects_.at(index).jets_ - (index==0? 0 : filterObjects_.at(index-1).jets_);
    }

    /// number of composites for a specific filter
    size_type numComposites(size_type index) const {
      return filterObjects_.at(index).composites_ - (index==0? 0 : filterObjects_.at(index-1).composites_);
    }

    /// number of mets for a specific filter
    size_type numMETs(size_type index) const {
      return filterObjects_.at(index).mets_ - (index==0? 0 : filterObjects_.at(index-1).mets_);
    }

    /// number of hts for a specific filter
    size_type numHTs(size_type index) const {
      return filterObjects_.at(index).hts_ - (index==0? 0 : filterObjects_.at(index-1).hts_);
    }

    /// number of others for a specific filter
    size_type numOthers(size_type index) const {
      return filterObjects_.at(index).others_ - (index==0? 0 : filterObjects_.at(index-1).others_);
    }


    /// photons: _begin and _end iterators for specific filter
    std::vector<reco::RecoEcalCandidateRef>::const_iterator photons_begin(size_type index) const {
      return photons_.begin() + (index==0? 0 : filterObjects_.at(index-1).photons_);
    }
    std::vector<reco::RecoEcalCandidateRef>::const_iterator photons_end(size_type index) const {
      return photons_.begin() + filterObjects_.at(index).photons_;
    }


    /// electrons: _begin and _end iterators for specific filter
    std::vector<reco::ElectronRef>::const_iterator electrons_begin(size_type index) const {
      return electrons_.begin() + (index==0? 0 : filterObjects_.at(index-1).electrons_);
    }
    std::vector<reco::ElectronRef>::const_iterator electrons_end(size_type index) const {
      return electrons_.begin() + filterObjects_.at(index).electrons_;
    }


    /// muons: _begin and _end iterators for specific filter
    std::vector<reco::RecoChargedCandidateRef>::const_iterator muons_begin(size_type index) const {
      return muons_.begin() + (index==0? 0 : filterObjects_.at(index-1).muons_);
    }
    std::vector<reco::RecoChargedCandidateRef>::const_iterator muons_end(size_type index) const {
      return muons_.begin() + filterObjects_.at(index).muons_;
    }


    /// jets: _begin and _end iterators for specific filter
    std::vector<reco::CaloJetRef>::const_iterator jets_begin(size_type index) const {
      return jets_.begin() + (index==0? 0 : filterObjects_.at(index-1).jets_);
    }
    std::vector<reco::CaloJetRef>::const_iterator jets_end(size_type index) const {
      return jets_.begin() + filterObjects_.at(index).jets_;
    }


    /// composites: _begin and _end iterators for specific filter
    std::vector<reco::CompositeCandidateRef>::const_iterator composites_begin(size_type index) const {
      return composites_.begin() + (index==0? 0 : filterObjects_.at(index-1).composites_);
    }
    std::vector<reco::CompositeCandidateRef>::const_iterator composites_end(size_type index) const {
      return composites_.begin() + filterObjects_.at(index).composites_;
    }


    /// mets: _begin and _end iterators for specific filter
    std::vector<reco::CaloMETRef>::const_iterator mets_begin(size_type index) const {
      return mets_.begin() + (index==0? 0 : filterObjects_.at(index-1).mets_);
    }
    std::vector<reco::CaloMETRef>::const_iterator mets_end(size_type index) const {
      return mets_.begin() + filterObjects_.at(index).mets_;
    }


    /// hts: _begin and _end iterators for specific filter
    std::vector<reco::METRef>::const_iterator hts_begin(size_type index) const {
      return hts_.begin() + (index==0? 0 : filterObjects_.at(index-1).hts_);
    }
    std::vector<reco::METRef>::const_iterator hts_end(size_type index) const {
      return hts_.begin() + filterObjects_.at(index).hts_;
    }


    /// others: _begin and _end iterators for specific filter
    std::vector<XRef>::const_iterator others_begin(size_type index) const {
      return others_.begin() + (index==0? 0 : filterObjects_.at(index-1).others_);
    }
    std::vector<XRef>::const_iterator others_end(size_type index) const {
      return others_.begin() + filterObjects_.at(index).others_;
    }


    /// get keys of objects passing specific filter in the collection identified by its ProductID
    void otherKeys(size_type index, edm::ProductID id, Keys& keys) const {
      keys.resize(0);
      const std::vector<XRef>::const_iterator begin(others_begin(index));
      const std::vector<XRef>::const_iterator end(others_end(index));
      for (std::vector<XRef>::const_iterator i=begin; i!=end; ++i) {
	if (i->first==id) keys.push_back(i->second);
      }
    }
    /// get vector of Ref<C> using handle to original collection for type
    template <typename C>
    void getOthers(size_type index, const edm::Handle<C>& handle, std::vector<edm::Ref<C> >& vref) const {
      Keys keys();
      otherKeys(index, handle.id(), keys);
      const size_type n(keys.size());
      vref.resize(n);
      for (size_type i=0; i!=n; ++i) {
	vref[i]=edm::Ref<C>(handle,keys[i]);
      }
      return;
    }

  };

}

#endif
