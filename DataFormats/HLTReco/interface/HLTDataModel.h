#ifndef HLTReco_HLTDataModel_h
#define HLTReco_HLTDataModel_h

/** \class reco::HLTDataModel
 *
 *  Classes for new HLT data model (to be split into separate header files)
 *
 *  $Date: 2007/11/26 16:51:16 $
 *  $Revision: 1.8 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/Math/interface/LorentzVector.h"

#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TauReco/interface/HLTTauFwd.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include <string>
#include <vector>

namespace trigger
{

  /// space saving index type for trigger (size_t is too large - 64 bits)
  typedef uint16_t tInd;

  /// 4-momentum of a trigger physics object
  typedef math::PtEtaPhiMLorentzVectorF TriggerFourMomentum;


  /// Single trigger physics object (e.g., an isolated muon)
  class TriggerObject {

  /// data members
  private:
    /// 4-momentum of physics object
    TriggerFourMomentum objectP4_;
    /// id or type - similar to pdgId
    int objectId_;

  /// methods
  public:
    /// constructors
    TriggerObject(): objectP4_(), objectId_() { }
    TriggerObject(const TriggerFourMomentum& P4, int Id=0): objectP4_(P4), objectId_(Id) { }

    /// setters
    void setP4 (const TriggerFourMomentum& P4) {objectP4_ = P4;}
    void setId (int Id=0) {objectId_ = Id;}

    /// getters
    const TriggerFourMomentum& getP4() const {return objectP4_;}
    int getId() const {return objectId_;}

  };


  /// collection of trigger physics objects (e.g., all isolated muons)
  typedef std::vector<TriggerObject> TriggerObjectCollection;


  /// Transient book-keeping EDProduct filled by HLTFilter modules to
  /// record physics objects firing the filter (not persistet in 
  /// production - same functionality but different implementation 
  /// compared to the old HLT data model's HLTFilterObjectWithRefs class)
  class TriggerFilterObjectWithRefs {

  /// data members
  private:
    /// label of filter module for which the info is recorded here
    /// (can be recovered from "provenance" of product instance? how?)
    std::string filterLabel_;
    /// Ref<C> more efficient than Ptr<T> - perhaps use RefVector?
    std::vector<edm::Ref<reco::PhotonCollection> > photons_;
    std::vector<edm::Ref<reco::ElectronCollection> > electrons_;
    std::vector<edm::Ref<reco::MuonCollection> > muons_;
    std::vector<edm::Ref<reco::HLTTauCollection> > taus_;
    std::vector<edm::Ref<reco::CaloJetCollection> > jets_;
    
  /// methods
  public:
    /// constructors
    TriggerFilterObjectWithRefs():
      filterLabel_(),
      photons_(), electrons_(), muons_(), taus_(), jets_() { }
    TriggerFilterObjectWithRefs(const std::string& filterLabel):
      filterLabel_(filterLabel),
      photons_(), electrons_(), muons_(), taus_(), jets_() { }

    /// setters
    void addObject(const edm::Ref<reco::PhotonCollection>& photon) {photons_.push_back(photon);}
    void addObject(const edm::Ref<reco::ElectronCollection>& electron) {electrons_.push_back(electron);}
    void addObject(const edm::Ref<reco::MuonCollection>& muon) {muons_.push_back(muon);}
    void addObject(const edm::Ref<reco::HLTTauCollection>& tau) {taus_.push_back(tau);}
    void addObject(const edm::Ref<reco::CaloJetCollection>& jet) {jets_.push_back(jet);}

    /// getters
    const std::string& getLabel() const {return filterLabel_;}

    const std::vector<edm::Ref<reco::PhotonCollection> >& getPhotons() const {return photons_;}
    const std::vector<edm::Ref<reco::ElectronCollection> >& getElectrons() const {return electrons_;}
    const std::vector<edm::Ref<reco::MuonCollection> >& getMuons() const {return muons_;}
    const std::vector<edm::Ref<reco::HLTTauCollection> >& getTaus() const {return taus_;}
    const std::vector<edm::Ref<reco::CaloJetCollection> >& getJets() const {return jets_;}

  };



  /// The single EDProduct to be saved for each event (AOD case)
  class TriggerEvent {

  private:

    /// Helper class: recording trigger objects firing a single filter
    class TriggerFilterObject {
    public:
      /// the label of the filter
      std::string filterLabel_;
      /// indices pointing into collection of unique TriggerObjects
      std::vector<tInd> filterKeys_;

      /// constructors
      TriggerFilterObject(): filterLabel_(), filterKeys_() { }
      TriggerFilterObject(const std::string& filterLabel): filterLabel_(filterLabel), filterKeys_() { }
      TriggerFilterObject(const std::string& filterLabel, const std::vector<tInd>& filterKeys): filterLabel_(filterLabel), filterKeys_(filterKeys) { }
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
    void addObject(const TriggerObject& triggerObject) {triggerObjects_.push_back(triggerObject);}
    void addFilter(const std::string& filterLabel, const std::vector<tInd>& keys) {triggerFilters_.push_back(TriggerFilterObject(filterLabel, keys));}

    /// getters
    const TriggerObjectCollection& getObjects() const {return triggerObjects_;}
    const TriggerObject& getObject(tInd index) const {return triggerObjects_.at(index);}
    const std::string& getFilterLabel(tInd index) const {return triggerFilters_.at(index).filterLabel_;}
    const std::vector<tInd>& getFilterKeys(tInd index) const {return triggerFilters_.at(index).filterKeys_;}

    /// other
    tInd numObjects() const {return triggerObjects_.size();}
    tInd numFilters() const {return triggerFilters_.size();}

  };



  /// The single EDProduct to be saved in addition for each event
  /// - but only in the RAW case
  class TriggerEventWithRefs {

  private:

    /// Helper class: trigger objects firing a single filter
    class TriggerFilterObject {
    public:
      /// label of filter
      std::string filterLabel_;
      /// end indices into linearised vector of Refs
      /// (-> first start index is always 0)
      tInd photons_;
      tInd electrons_;
      tInd muons_;
      tInd taus_;
      tInd jets_;
      /// constructor
      TriggerFilterObject() :
	filterLabel_(),
	photons_(0), electrons_(0), muons_(0), taus_(0), jets_(0) { }
      TriggerFilterObject(const std::string& filterLabel,
          tInd np, tInd ne, tInd nm, tInd nt, tInd nj) :
	filterLabel_(filterLabel),
	photons_(np), electrons_(ne), muons_(nm), taus_(nt), jets_(nj) { }
    };

  /// data members
  private:
    /// the filters recorded here
    std::vector<TriggerFilterObject> filterObjects_;
    /// non-owning pointers into collections (linearised)
    std::vector<edm::Ref<reco::PhotonCollection> > photons_;
    std::vector<edm::Ref<reco::ElectronCollection> > electrons_;
    std::vector<edm::Ref<reco::MuonCollection> > muons_;
    std::vector<edm::Ref<reco::HLTTauCollection> > taus_;
    std::vector<edm::Ref<reco::CaloJetCollection> > jets_;
   
  /// methods
  public:
    /// constructors
    TriggerEventWithRefs(): filterObjects_(),
      photons_(), electrons_(), muons_(), taus_(), jets_() { }

    /// setters - to build EDProduct
    void addObject(const TriggerFilterObjectWithRefs& tfowr) {
      photons_.insert(photons_.end(),tfowr.getPhotons().begin(),tfowr.getPhotons().end());
      electrons_.insert(electrons_.end(),tfowr.getElectrons().begin(),tfowr.getElectrons().end());
      muons_.insert(muons_.end(),tfowr.getMuons().begin(),tfowr.getMuons().end());
      taus_.insert(taus_.end(),tfowr.getTaus().begin(),tfowr.getTaus().end());
      jets_.insert(jets_.end(),tfowr.getJets().begin(),tfowr.getJets().end());
      filterObjects_.push_back(
        TriggerFilterObject (tfowr.getLabel(), 
			    photons_.size(), electrons_.size(), 
			    muons_.size(), taus_.size(), jets_.size()
			    )
	);
    }

    /// getters - for user access
    tInd numFilters() const {return filterObjects_.size();}

    const std::string& getFilterLabel(tInd key) const {return filterObjects_.at(key).filterLabel_;}


    tInd find(const std::string& filterLabel) const {
      for (tInd i=0; i!=numFilters(); ++i) {
	if (filterLabel==filterObjects_[i].filterLabel_) {return i;}
      }
      return numFilters();
    }

    tInd numPhotons(tInd key) const {
      return filterObjects_.at(key).photons_ - (key==0? 0 : filterObjects_.at(key-1).photons_);
    }
    tInd numPhotons(const std::string& filterLabel) const {
      return numPhotons(find(filterLabel));
    }

    tInd numElectrons(tInd key) const {
      return filterObjects_.at(key).electrons_ - (key==0? 0 : filterObjects_.at(key-1).electrons_);
    }
    tInd numElectrons(const std::string& filterLabel) const {
      return numElectrons(find(filterLabel));
    }

    tInd numMuons(tInd key) const {
      return filterObjects_.at(key).muons_ - (key==0? 0 : filterObjects_.at(key-1).muons_);
    }
    tInd numMuons(const std::string& filterLabel) const {
      return numMuons(find(filterLabel));
    }

    tInd numTaus(tInd key) const {
      return filterObjects_.at(key).taus_ - (key==0? 0 : filterObjects_.at(key-1).taus_);
    }
    tInd numTaus(const std::string& filterLabel) const {
      return numTaus(find(filterLabel));
    }

    tInd numJets(tInd key) const {
      return filterObjects_.at(key).jets_ - (key==0? 0 : filterObjects_.at(key-1).jets_);
    }
    tInd numJets(const std::string& filterLabel) const {
      return numJets(find(filterLabel));
    }

    /// iterators

    std::vector<edm::Ref<reco::PhotonCollection> >::const_iterator
      photons_begin(tInd key) const
    { return photons_.begin() + 
      (key==0? 0 : filterObjects_.at(key-1).photons_);
    }
    std::vector<edm::Ref<reco::PhotonCollection> >::const_iterator
      photons_end(tInd key) const
    { return photons_.begin() + filterObjects_.at(key).photons_; }


    std::vector<edm::Ref<reco::ElectronCollection> >::const_iterator
      electrons_begin(tInd key) const
    { return electrons_.begin() + 
      (key==0? 0 : filterObjects_[key-1].electrons_);
    }
    std::vector<edm::Ref<reco::ElectronCollection> >::const_iterator
      electrons_end(tInd key) const
    { return electrons_.begin() + filterObjects_[key].electrons_; }


    std::vector<edm::Ref<reco::MuonCollection> >::const_iterator
      muons_begin(tInd key) const
    { return muons_.begin() + 
      (key==0? 0 : filterObjects_[key-1].muons_);
    }
    std::vector<edm::Ref<reco::MuonCollection> >::const_iterator
      muons_end(tInd key) const
    { return muons_.begin() + filterObjects_[key].muons_; }


    std::vector<edm::Ref<reco::HLTTauCollection> >::const_iterator
      taus_begin(tInd key) const
    { return taus_.begin() + 
      (key==0? 0 : filterObjects_[key-1].taus_);
    }
    std::vector<edm::Ref<reco::HLTTauCollection> >::const_iterator
      taus_end(tInd key) const
    { return taus_.begin() + filterObjects_[key].taus_; }


    std::vector<edm::Ref<reco::CaloJetCollection> >::const_iterator
      jets_begin(tInd key) const
    { return jets_.begin() + 
      (key==0? 0 : filterObjects_[key-1].jets_);
    }
    std::vector<edm::Ref<reco::CaloJetCollection> >::const_iterator
      jets_end(tInd key) const
    { return jets_.begin() + filterObjects_[key].jets_; }


  };

}

#endif
