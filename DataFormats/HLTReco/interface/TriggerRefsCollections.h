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
 *  $Date: 2008/01/12 16:53:54 $
 *  $Revision: 1.7 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Candidate/interface/CompositeCandidateFwd.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"

#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"

#include <cassert>
#include <vector>

#include<typeinfo>

namespace trigger
{

  typedef std::vector<reco::RecoEcalCandidateRef>           VRphoton;
  typedef std::vector<reco::ElectronRef>                    VRelectron;
  typedef std::vector<reco::RecoChargedCandidateRef>        VRmuon;
  typedef std::vector<reco::CaloJetRef>                     VRjet;
  typedef std::vector<reco::CompositeCandidateRef>          VRcomposite;
  typedef std::vector<reco::CaloMETRef>                     VRmet;
  typedef std::vector<reco::METRef>                         VRht;
  typedef std::vector<reco::IsolatedPixelTrackCandidateRef> VRpixtrack;

  typedef std::vector<l1extra::L1EmParticleRef>             VRl1em;
  typedef std::vector<l1extra::L1MuonParticleRef>           VRl1muon;
  typedef std::vector<l1extra::L1JetParticleRef>            VRl1jet;
  typedef std::vector<l1extra::L1EtMissParticleRef>         VRl1etmiss;

  class TriggerRefsCollections {

  /// data members
  private:
    /// physics type ids and Refs
    Vids        photonIds_;
    VRphoton    photonRefs_;
    Vids        electronIds_;
    VRelectron  electronRefs_;
    Vids        muonIds_;
    VRmuon      muonRefs_;
    Vids        jetIds_;
    VRjet       jetRefs_;
    Vids        compositeIds_;
    VRcomposite compositeRefs_;
    Vids        metIds_;
    VRmet       metRefs_;
    Vids        htIds_;
    VRht        htRefs_;
    Vids        pixtrackIds_;
    VRpixtrack  pixtrackRefs_;

    Vids        l1emIds_;
    VRl1em      l1emRefs_;
    Vids        l1muonIds_;
    VRl1muon    l1muonRefs_;
    Vids        l1jetIds_;
    VRl1jet     l1jetRefs_;
    Vids        l1etmissIds_;
    VRl1etmiss  l1etmissRefs_;
    
  /// methods
  public:
    /// constructors
    TriggerRefsCollections() :
      photonIds_(), photonRefs_(),
      electronIds_(), electronRefs_(),
      muonIds_(), muonRefs_(),
      compositeIds_(), compositeRefs_(),
      metIds_(), metRefs_(),
      htIds_(), htRefs_(),
      pixtrackIds_(), pixtrackRefs_(),

      l1emIds_(), l1emRefs_(),
      l1muonIds_(), l1muonRefs_(),
      l1jetIds_(), l1jetRefs_(),
      l1etmissIds_(), l1etmissRefs_()
      { }

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
    void addObject(int id, const reco::IsolatedPixelTrackCandidateRef& ref) {
      pixtrackIds_.push_back(id);
      pixtrackRefs_.push_back(ref);
    }

    void addObject(int id, const l1extra::L1EmParticleRef& ref) {
      l1emIds_.push_back(id);
      l1emRefs_.push_back(ref);
    }
    void addObject(int id, const l1extra::L1MuonParticleRef& ref) {
      l1muonIds_.push_back(id);
      l1muonRefs_.push_back(ref);
    }
    void addObject(int id, const l1extra::L1JetParticleRef& ref) {
      l1jetIds_.push_back(id);
      l1jetRefs_.push_back(ref);
    }
    void addObject(int id, const l1extra::L1EtMissParticleRef& ref) {
      l1etmissIds_.push_back(id);
      l1etmissRefs_.push_back(ref);
    }


    /// 
    size_type addObjects (const Vids& ids, const VRphoton& refs) {
      assert(ids.size()==refs.size());
      photonIds_.insert(photonIds_.end(),ids.begin(),ids.end());
      photonRefs_.insert(photonRefs_.end(),refs.begin(),refs.end());
      return photonIds_.size();
    }
    size_type addObjects (const Vids& ids, const VRelectron& refs) {
      assert(ids.size()==refs.size());
      electronIds_.insert(electronIds_.end(),ids.begin(),ids.end());
      electronRefs_.insert(electronRefs_.end(),refs.begin(),refs.end());
      return electronIds_.size();
    }
    size_type addObjects (const Vids& ids, const VRmuon& refs) {
      assert(ids.size()==refs.size());
      muonIds_.insert(muonIds_.end(),ids.begin(),ids.end());
      muonRefs_.insert(muonRefs_.end(),refs.begin(),refs.end());
      return muonIds_.size();
    }
    size_type addObjects (const Vids& ids, const VRjet& refs) {
      assert(ids.size()==refs.size());
      jetIds_.insert(jetIds_.end(),ids.begin(),ids.end());
      jetRefs_.insert(jetRefs_.end(),refs.begin(),refs.end());
      return jetIds_.size();
    }
    size_type addObjects (const Vids& ids, const VRcomposite& refs) {
      assert(ids.size()==refs.size());
      compositeIds_.insert(compositeIds_.end(),ids.begin(),ids.end());
      compositeRefs_.insert(compositeRefs_.end(),refs.begin(),refs.end());
      return compositeIds_.size();
    }
    size_type addObjects (const Vids& ids, const VRmet& refs) {
      assert(ids.size()==refs.size());
      metIds_.insert(metIds_.end(),ids.begin(),ids.end());
      metRefs_.insert(metRefs_.end(),refs.begin(),refs.end());
      return metIds_.size();
    }
    size_type addObjects (const Vids& ids, const VRht& refs) {
      assert(ids.size()==refs.size());
      htIds_.insert(htIds_.end(),ids.begin(),ids.end());
      htRefs_.insert(htRefs_.end(),refs.begin(),refs.end());
      return htIds_.size();
    }
    size_type addObjects (const Vids& ids, const VRpixtrack& refs) {
      assert(ids.size()==refs.size());
      pixtrackIds_.insert(pixtrackIds_.end(),ids.begin(),ids.end());
      pixtrackRefs_.insert(pixtrackRefs_.end(),refs.begin(),refs.end());
      return pixtrackIds_.size();
    }

    size_type addObjects (const Vids& ids, const VRl1em& refs) {
      assert(ids.size()==refs.size());
      l1emIds_.insert(l1emIds_.end(),ids.begin(),ids.end());
      l1emRefs_.insert(l1emRefs_.end(),refs.begin(),refs.end());
      return l1emIds_.size();
    }
    size_type addObjects (const Vids& ids, const VRl1muon& refs) {
      assert(ids.size()==refs.size());
      l1muonIds_.insert(l1muonIds_.end(),ids.begin(),ids.end());
      l1muonRefs_.insert(l1muonRefs_.end(),refs.begin(),refs.end());
      return l1muonIds_.size();
    }
    size_type addObjects (const Vids& ids, const VRl1jet& refs) {
      assert(ids.size()==refs.size());
      l1jetIds_.insert(l1jetIds_.end(),ids.begin(),ids.end());
      l1jetRefs_.insert(l1jetRefs_.end(),refs.begin(),refs.end());
      return l1jetIds_.size();
    }
    size_type addObjects (const Vids& ids, const VRl1etmiss& refs) {
      assert(ids.size()==refs.size());
      l1etmissIds_.insert(l1etmissIds_.end(),ids.begin(),ids.end());
      l1etmissRefs_.insert(l1etmissRefs_.end(),refs.begin(),refs.end());
      return l1emIds_.size();
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
	if (id==htIds_[i]) {refs[j]=htRefs_[i]; ++j;}
      }
      return;
    }
    void getObjects(int id, VRpixtrack& refs) const {
      getObjects(id,refs,0,pixtrackIds_.size());
    } 
    void getObjects(int id, VRpixtrack& refs, size_type begin, size_type end) const {
      size_type n(0);
      for (size_type i=begin; i!=end; ++i) {if (id==pixtrackIds_[i]) {++n;}}
      refs.resize(n);
      size_type j(0);
      for (size_type i=begin; i!=end; ++i) {
	if (id==pixtrackIds_[i]) {refs[j]=pixtrackRefs_[i]; ++j;}
      }
      return;
    }

    void getObjects(int id, VRl1em& refs) const {
      getObjects(id,refs,0,l1emIds_.size());
    } 
    void getObjects(int id, VRl1em& refs, size_type begin, size_type end) const {
      size_type n(0);
      for (size_type i=begin; i!=end; ++i) {if (id==l1emIds_[i]) {++n;}}
      refs.resize(n);
      size_type j(0);
      for (size_type i=begin; i!=end; ++i) {
	if (id==l1emIds_[i]) {refs[j]=l1emRefs_[i]; ++j;}
      }
      return;
    }
    void getObjects(int id, VRl1muon& refs) const {
      getObjects(id,refs,0,l1muonIds_.size());
    } 
    void getObjects(int id, VRl1muon& refs, size_type begin, size_type end) const {
      size_type n(0);
      for (size_type i=begin; i!=end; ++i) {if (id==l1muonIds_[i]) {++n;}}
      refs.resize(n);
      size_type j(0);
      for (size_type i=begin; i!=end; ++i) {
	if (id==l1muonIds_[i]) {refs[j]=l1muonRefs_[i]; ++j;}
      }
      return;
    }
    void getObjects(int id, VRl1jet& refs) const {
      getObjects(id,refs,0,l1jetIds_.size());
    } 
    void getObjects(int id, VRl1jet& refs, size_type begin, size_type end) const {
      size_type n(0);
      for (size_type i=begin; i!=end; ++i) {if (id==l1jetIds_[i]) {++n;}}
      refs.resize(n);
      size_type j(0);
      for (size_type i=begin; i!=end; ++i) {
	if (id==l1jetIds_[i]) {refs[j]=l1jetRefs_[i]; ++j;}
      }
      return;
    }
    void getObjects(int id, VRl1etmiss& refs) const {
      getObjects(id,refs,0,l1etmissIds_.size());
    } 
    void getObjects(int id, VRl1etmiss& refs, size_type begin, size_type end) const {
      size_type n(0);
      for (size_type i=begin; i!=end; ++i) {if (id==l1etmissIds_[i]) {++n;}}
      refs.resize(n);
      size_type j(0);
      for (size_type i=begin; i!=end; ++i) {
	if (id==l1etmissIds_[i]) {refs[j]=l1etmissRefs_[i]; ++j;}
      }
      return;
    }


    /// low-level getters for data members
    const Vids&        photonIds()     const {return photonIds_;}
    const VRphoton&    photonRefs()    const {return photonRefs_;}
    const Vids&        electronIds()   const {return electronIds_;}
    const VRelectron&  electronRefs()  const {return electronRefs_;}
    const Vids&        muonIds()       const {return muonIds_;}
    const VRmuon&      muonRefs()      const {return muonRefs_;}
    const Vids&        jetIds()        const {return jetIds_;}
    const VRjet&       jetRefs()       const {return jetRefs_;}
    const Vids&        compositeIds()  const {return compositeIds_;}
    const VRcomposite& compositeRefs() const {return compositeRefs_;}
    const Vids&        metIds()        const {return metIds_;}
    const VRmet&       metRefs()       const {return metRefs_;}
    const Vids&        htIds()         const {return htIds_;}
    const VRht&        htRefs()        const {return htRefs_;}
    const Vids&        pixtrackIds()   const {return pixtrackIds_;}
    const VRpixtrack&  pixtrackRefs()  const {return pixtrackRefs_;}

    const Vids&        l1emIds()       const {return l1emIds_;}
    const VRl1em&      l1emRefs()      const {return l1emRefs_;}
    const Vids&        l1muonIds()     const {return l1muonIds_;}
    const VRl1muon&    l1muonRefs()    const {return l1muonRefs_;}
    const Vids&        l1jetIds()      const {return l1jetIds_;}
    const VRl1jet&     l1jetRefs()     const {return l1jetRefs_;}
    const Vids&        l1etmissIds()   const {return l1etmissIds_;}
    const VRl1etmiss&  l1etmissRefs()  const {return l1etmissRefs_;}

  };

}

#endif
