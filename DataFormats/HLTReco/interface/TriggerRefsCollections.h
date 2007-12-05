#ifndef HLTReco_TriggerRefsCollections_h
#define HLTReco_TriggerRefsCollections_h

/** \class trigger::TriggerRefsCollections
 *
 *  Holds the collections of Ref<C>s which describe the physics
 *  objects passing trigger cuts.
 *
 *  This implementation is not completely space-efficient as some
 *  physics object containers may stay empty. However, the big
 *  advantage is that the solution is generic, i.e., works for all
 *  possible HLT filters. Hence we accept the reasonably small
 *  overhead of empty containers.
 *
 *  $Date: 2007/12/04 20:40:37 $
 *  $Revision: 1.8 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Candidate/interface/CompositeCandidateFwd.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/METReco/interface/METFwd.h"

#include <cassert>
#include <utility>
#include <vector>

namespace trigger
{

  typedef std::vector<int>                           Vints;
  typedef std::vector<reco::RecoEcalCandidateRef>    VRphotons;
  typedef std::vector<reco::ElectronRef>             VRelectrons;
  typedef std::vector<reco::RecoChargedCandidateRef> VRmuons;
  typedef std::vector<reco::CaloJetRef>              VRjets;
  typedef std::vector<reco::CompositeCandidateRef>   VRcomposites;
  typedef std::vector<reco::CaloMETRef>              VRmets;
  typedef std::vector<reco::METRef>                  VRhts;

  class TriggerRefsCollections {

  /// data members
  private:
    /// physics type ids and Refs
    Vints        photonIds_;
    VRphotons    photonRefs_;
    Vints        electronIds_;
    VRelectrons  electronRefs_;
    Vints        muonIds_;
    VRmuons      muonRefs_;
    Vints        jetIds_;
    VRjets       jetRefs_;
    Vints        compositeIds_;
    VRcomposites compositeRefs_;
    Vints        metIds_;
    VRmets       metRefs_;
    Vints        htIds_;
    VRhts        htRefs_;
    
  /// methods
  public:
    /// constructors
    TriggerRefsCollections() :
      photonIds_(), photonRefs_(),
      electronIds_(), electronRefs_(),
      muonIds_(), muonRefs_(),
      compositeIds_(), compositeRefs_(),
      metIds_(), metRefs_(),
      htIds_(), htRefs_() { }

    /// setters for L3 collections: (id=physics type, and Ref<C>)
    void addObject(int id, const reco::RecoEcalCandidateRef& ref) {
      photonIds_.push_back(id);
      photonRefs_.push_back(ref);
    }
    void addObject(int id, const reco::ElectronRef& ref) {
      electronIds_.push_back(id);
      electronRefs_.push_back(ref);
    }
    void addObject(int id, const reco::RecoChargedCandidateRef& ref) {
      muonIds_.push_back(id);
      muonRefs_.push_back(ref);
    }
    void addObject(int id, const reco::CaloJetRef& ref) {
      jetIds_.push_back(id);
      jetRefs_.push_back(ref);
    }
    void addObject(int id, const reco::CompositeCandidateRef& ref) {
      compositeIds_.push_back(id);
      compositeRefs_.push_back(ref);
    }
    void addObject(int id, const reco::CaloMETRef& ref) {
      metIds_.push_back(id);
      metRefs_.push_back(ref);
    }
    void addObject(int id, const reco::METRef& ref) {
      htIds_.push_back(id);
      htRefs_.push_back(ref);
    }

    /// 
    size_type append (const Vints& ids, const VRphotons& refs) {
      assert(ids.size()==refs.size());
      photonIds_.insert(photonIds_.end(),ids.begin(),ids.end());
      photonRefs_.insert(photonRefs_.end(),refs.begin(),refs.end());
      return photonIds_.size();
    }
    size_type append (const Vints& ids, const VRelectrons& refs) {
      assert(ids.size()==refs.size());
      electronIds_.insert(electronIds_.end(),ids.begin(),ids.end());
      electronRefs_.insert(electronRefs_.end(),refs.begin(),refs.end());
      return electronIds_.size();
    }
    size_type append (const Vints& ids, const VRmuons& refs) {
      assert(ids.size()==refs.size());
      muonIds_.insert(muonIds_.end(),ids.begin(),ids.end());
      muonRefs_.insert(muonRefs_.end(),refs.begin(),refs.end());
      return muonIds_.size();
    }
    size_type append (const Vints& ids, const VRjets& refs) {
      assert(ids.size()==refs.size());
      jetIds_.insert(jetIds_.end(),ids.begin(),ids.end());
      jetRefs_.insert(jetRefs_.end(),refs.begin(),refs.end());
      return jetIds_.size();
    }
    size_type append (const Vints& ids, const VRcomposites& refs) {
      assert(ids.size()==refs.size());
      compositeIds_.insert(compositeIds_.end(),ids.begin(),ids.end());
      compositeRefs_.insert(compositeRefs_.end(),refs.begin(),refs.end());
      return compositeIds_.size();
    }
    size_type append (const Vints& ids, const VRmets& refs) {
      assert(ids.size()==refs.size());
      metIds_.insert(metIds_.end(),ids.begin(),ids.end());
      metRefs_.insert(metRefs_.end(),refs.begin(),refs.end());
      return metIds_.size();
    }
    size_type append (const Vints& ids, const VRhts& refs) {
      assert(ids.size()==refs.size());
      htIds_.insert(htIds_.end(),ids.begin(),ids.end());
      htRefs_.insert(htRefs_.end(),refs.begin(),refs.end());
      return htIds_.size();
    }

    /// physics-level getters: get Ref<C>s for physics type id within a slice
    void getObjects(int id, VRphotons& refs) const {
      getObjects(id,refs,0,photonIds_.size());
    }
    void getObjects(int id, VRphotons& refs, size_type begin, size_type end) const {
      size_type n(0);
      for (size_type i=begin; i!=end; ++i) {if (id==photonIds_[i]) {++n;}}
      refs.resize(n);
      size_type j(0);
      for (size_type i=begin; i!=end; ++i) {
	if (id==photonIds_[i]) {refs[j]=photonRefs_[i]; ++j;}
      }
      return;
    }
    void getObjects(int id, VRelectrons& refs) const {
      getObjects(id,refs,0,electronIds_.size());
    }
    void getObjects(int id, VRelectrons& refs, size_type begin, size_type end) const {
      size_type n(0);
      for (size_type i=begin; i!=end; ++i) {if (id==electronIds_[i]) {++n;}}
      refs.resize(n);
      size_type j(0);
      for (size_type i=begin; i!=end; ++i) {
	if (id==electronIds_[i]) {refs[j]=electronRefs_[i]; ++j;}
      }
      return;
    }
    void getObjects(int id, VRmuons& refs) const {
      getObjects(id,refs,0,muonIds_.size());
    }
    void getObjects(int id, VRmuons& refs, size_type begin, size_type end) const {
      size_type n(0);
      for (size_type i=begin; i!=end; ++i) {if (id==muonIds_[i]) {++n;}}
      refs.resize(n);
      size_type j(0);
      for (size_type i=begin; i!=end; ++i) {
	if (id==muonIds_[i]) {refs[j]=muonRefs_[i]; ++j;}
      }
      return;
    }
    void getObjects(int id, VRjets& refs) const {
      getObjects(id,refs,0,jetIds_.size());
    }
    void getObjects(int id, VRjets& refs, size_type begin, size_type end) const {
      size_type n(0);
      for (size_type i=begin; i!=end; ++i) {if (id==jetIds_[i]) {++n;}}
      refs.resize(n);
      size_type j(0);
      for (size_type i=begin; i!=end; ++i) {
	if (id==jetIds_[i]) {refs[j]=jetRefs_[i]; ++j;}
      }
      return;
    }
    void getObjects(int id, VRcomposites& refs) const {
      getObjects(id,refs,0,compositeIds_.size());
    }
    void getObjects(int id, VRcomposites& refs, size_type begin, size_type end) const {
      size_type n(0);
      for (size_type i=begin; i!=end; ++i) {if (id==compositeIds_[i]) {++n;}}
      refs.resize(n);
      size_type j(0);
      for (size_type i=begin; i!=end; ++i) {
	if (id==compositeIds_[i]) {refs[j]=compositeRefs_[i]; ++j;}
      }
      return;
    }
    void getObjects(int id, VRmets& refs) const {
      getObjects(id,refs,0,metIds_.size());
    }
    void getObjects(int id, VRmets& refs, size_type begin, size_type end) const {
      size_type n(0);
      for (size_type i=begin; i!=end; ++i) {if (id==metIds_[i]) {++n;}}
      refs.resize(n);
      size_type j(0);
      for (size_type i=begin; i!=end; ++i) {
	if (id==metIds_[i]) {refs[j]=metRefs_[i]; ++j;}
      }
      return;
    }
    void getObjects(int id, VRhts& refs) const {
      getObjects(id,refs,0,htIds_.size());
    } 
    void getObjects(int id, VRhts& refs, size_type begin, size_type end) const {
      size_type n(0);
      for (size_type i=begin; i!=end; ++i) {if (id==htIds_[i]) {++n;}}
      refs.resize(n);
      size_type j(0);
      for (size_type i=begin; i!=end; ++i) {
	if (id==htIds_[i]) {refs[i]=htRefs_[i]; ++j;}
      }
      return;
    }

    /// low-level getters for data members
    const Vints&        photonIds()     const {return photonIds_;}
    const VRphotons&    photonRefs()    const {return photonRefs_;}
    const Vints&        electronIds()   const {return electronIds_;}
    const VRelectrons&  electronRefs()  const {return electronRefs_;}
    const Vints&        muonIds()       const {return muonIds_;}
    const VRmuons&      muonRefs()      const {return muonRefs_;}
    const Vints&        jetIds()        const {return jetIds_;}
    const VRjets&       jetRefs()       const {return jetRefs_;}
    const Vints&        compositeIds()  const {return compositeIds_;}
    const VRcomposites& compositeRefs() const {return compositeRefs_;}
    const Vints&        metIds()        const {return metIds_;}
    const VRmets&       metRefs()       const {return metRefs_;}
    const Vints&        htIds()         const {return htIds_;}
    const VRhts&        htRefs()        const {return htRefs_;}

  };

}

#endif
