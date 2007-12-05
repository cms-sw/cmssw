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
 *  $Date: 2007/12/05 14:24:02 $
 *  $Revision: 1.1 $
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

  typedef std::vector<int>                           Vint;

  typedef std::vector<reco::RecoEcalCandidateRef>    VRphoton;
  typedef std::vector<reco::ElectronRef>             VRelectron;
  typedef std::vector<reco::RecoChargedCandidateRef> VRmuon;
  typedef std::vector<reco::CaloJetRef>              VRjet;
  typedef std::vector<reco::CompositeCandidateRef>   VRcomposite;
  typedef std::vector<reco::CaloMETRef>              VRmet;
  typedef std::vector<reco::METRef>                  VRht;

  class TriggerRefsCollections {

  /// data members
  private:
    /// physics type ids and Refs
    Vint        photonIds_;
    VRphoton    photonRefs_;
    Vint        electronIds_;
    VRelectron  electronRefs_;
    Vint        muonIds_;
    VRmuon      muonRefs_;
    Vint        jetIds_;
    VRjet       jetRefs_;
    Vint        compositeIds_;
    VRcomposite compositeRefs_;
    Vint        metIds_;
    VRmet       metRefs_;
    Vint        htIds_;
    VRht        htRefs_;
    
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
    size_type append (const Vint& ids, const VRphoton& refs) {
      assert(ids.size()==refs.size());
      photonIds_.insert(photonIds_.end(),ids.begin(),ids.end());
      photonRefs_.insert(photonRefs_.end(),refs.begin(),refs.end());
      return photonIds_.size();
    }
    size_type append (const Vint& ids, const VRelectron& refs) {
      assert(ids.size()==refs.size());
      electronIds_.insert(electronIds_.end(),ids.begin(),ids.end());
      electronRefs_.insert(electronRefs_.end(),refs.begin(),refs.end());
      return electronIds_.size();
    }
    size_type append (const Vint& ids, const VRmuon& refs) {
      assert(ids.size()==refs.size());
      muonIds_.insert(muonIds_.end(),ids.begin(),ids.end());
      muonRefs_.insert(muonRefs_.end(),refs.begin(),refs.end());
      return muonIds_.size();
    }
    size_type append (const Vint& ids, const VRjet& refs) {
      assert(ids.size()==refs.size());
      jetIds_.insert(jetIds_.end(),ids.begin(),ids.end());
      jetRefs_.insert(jetRefs_.end(),refs.begin(),refs.end());
      return jetIds_.size();
    }
    size_type append (const Vint& ids, const VRcomposite& refs) {
      assert(ids.size()==refs.size());
      compositeIds_.insert(compositeIds_.end(),ids.begin(),ids.end());
      compositeRefs_.insert(compositeRefs_.end(),refs.begin(),refs.end());
      return compositeIds_.size();
    }
    size_type append (const Vint& ids, const VRmet& refs) {
      assert(ids.size()==refs.size());
      metIds_.insert(metIds_.end(),ids.begin(),ids.end());
      metRefs_.insert(metRefs_.end(),refs.begin(),refs.end());
      return metIds_.size();
    }
    size_type append (const Vint& ids, const VRht& refs) {
      assert(ids.size()==refs.size());
      htIds_.insert(htIds_.end(),ids.begin(),ids.end());
      htRefs_.insert(htRefs_.end(),refs.begin(),refs.end());
      return htIds_.size();
    }

    /// physics-level getters: get Ref<C>s for physics type id within a slice
    void getObjects(int id, VRphoton& refs) const {
      getObjects(id,refs,0,photonIds_.size());
    }
    void getObjects(int id, VRphoton& refs, size_type begin, size_type end) const {
      size_type n(0);
      for (size_type i=begin; i!=end; ++i) {if (id==photonIds_[i]) {++n;}}
      refs.resize(n);
      size_type j(0);
      for (size_type i=begin; i!=end; ++i) {
	if (id==photonIds_[i]) {refs[j]=photonRefs_[i]; ++j;}
      }
      return;
    }
    void getObjects(int id, VRelectron& refs) const {
      getObjects(id,refs,0,electronIds_.size());
    }
    void getObjects(int id, VRelectron& refs, size_type begin, size_type end) const {
      size_type n(0);
      for (size_type i=begin; i!=end; ++i) {if (id==electronIds_[i]) {++n;}}
      refs.resize(n);
      size_type j(0);
      for (size_type i=begin; i!=end; ++i) {
	if (id==electronIds_[i]) {refs[j]=electronRefs_[i]; ++j;}
      }
      return;
    }
    void getObjects(int id, VRmuon& refs) const {
      getObjects(id,refs,0,muonIds_.size());
    }
    void getObjects(int id, VRmuon& refs, size_type begin, size_type end) const {
      size_type n(0);
      for (size_type i=begin; i!=end; ++i) {if (id==muonIds_[i]) {++n;}}
      refs.resize(n);
      size_type j(0);
      for (size_type i=begin; i!=end; ++i) {
	if (id==muonIds_[i]) {refs[j]=muonRefs_[i]; ++j;}
      }
      return;
    }
    void getObjects(int id, VRjet& refs) const {
      getObjects(id,refs,0,jetIds_.size());
    }
    void getObjects(int id, VRjet& refs, size_type begin, size_type end) const {
      size_type n(0);
      for (size_type i=begin; i!=end; ++i) {if (id==jetIds_[i]) {++n;}}
      refs.resize(n);
      size_type j(0);
      for (size_type i=begin; i!=end; ++i) {
	if (id==jetIds_[i]) {refs[j]=jetRefs_[i]; ++j;}
      }
      return;
    }
    void getObjects(int id, VRcomposite& refs) const {
      getObjects(id,refs,0,compositeIds_.size());
    }
    void getObjects(int id, VRcomposite& refs, size_type begin, size_type end) const {
      size_type n(0);
      for (size_type i=begin; i!=end; ++i) {if (id==compositeIds_[i]) {++n;}}
      refs.resize(n);
      size_type j(0);
      for (size_type i=begin; i!=end; ++i) {
	if (id==compositeIds_[i]) {refs[j]=compositeRefs_[i]; ++j;}
      }
      return;
    }
    void getObjects(int id, VRmet& refs) const {
      getObjects(id,refs,0,metIds_.size());
    }
    void getObjects(int id, VRmet& refs, size_type begin, size_type end) const {
      size_type n(0);
      for (size_type i=begin; i!=end; ++i) {if (id==metIds_[i]) {++n;}}
      refs.resize(n);
      size_type j(0);
      for (size_type i=begin; i!=end; ++i) {
	if (id==metIds_[i]) {refs[j]=metRefs_[i]; ++j;}
      }
      return;
    }
    void getObjects(int id, VRht& refs) const {
      getObjects(id,refs,0,htIds_.size());
    } 
    void getObjects(int id, VRht& refs, size_type begin, size_type end) const {
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
    const Vint&        photonIds()     const {return photonIds_;}
    const VRphoton&    photonRefs()    const {return photonRefs_;}
    const Vint&        electronIds()   const {return electronIds_;}
    const VRelectron&  electronRefs()  const {return electronRefs_;}
    const Vint&        muonIds()       const {return muonIds_;}
    const VRmuon&      muonRefs()      const {return muonRefs_;}
    const Vint&        jetIds()        const {return jetIds_;}
    const VRjet&       jetRefs()       const {return jetRefs_;}
    const Vint&        compositeIds()  const {return compositeIds_;}
    const VRcomposite& compositeRefs() const {return compositeRefs_;}
    const Vint&        metIds()        const {return metIds_;}
    const VRmet&       metRefs()       const {return metRefs_;}
    const Vint&        htIds()         const {return htIds_;}
    const VRht&        htRefs()        const {return htRefs_;}

  };

}

#endif
