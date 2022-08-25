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
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/METReco/interface/PFMETFwd.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"

#include "DataFormats/L1Trigger/interface/L1HFRingsFwd.h"         // deprecate
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"      // deprecate
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"     // deprecate
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"    // deprecate
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"  // deprecate

#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/MuonShower.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"
#include "DataFormats/L1TCorrelator/interface/TkElectron.h"
#include "DataFormats/L1TCorrelator/interface/TkElectronFwd.h"
#include "DataFormats/L1TCorrelator/interface/TkEm.h"
#include "DataFormats/L1TCorrelator/interface/TkEmFwd.h"
#include "DataFormats/L1TParticleFlow/interface/PFJet.h"
#include "DataFormats/L1TParticleFlow/interface/PFTau.h"
#include "DataFormats/L1TParticleFlow/interface/HPSPFTau.h"
#include "DataFormats/L1TParticleFlow/interface/HPSPFTauFwd.h"

#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"

#include <cassert>
#include <vector>

#include <typeinfo>

namespace trigger {

  typedef std::vector<reco::RecoEcalCandidateRef> VRphoton;
  typedef std::vector<reco::ElectronRef> VRelectron;
  typedef std::vector<reco::RecoChargedCandidateRef> VRmuon;
  typedef std::vector<reco::CaloJetRef> VRjet;
  typedef std::vector<reco::CompositeCandidateRef> VRcomposite;
  typedef std::vector<reco::METRef> VRbasemet;
  typedef std::vector<reco::CaloMETRef> VRcalomet;
  typedef std::vector<reco::IsolatedPixelTrackCandidateRef> VRpixtrack;

  typedef std::vector<l1extra::L1EmParticleRef> VRl1em;          //deprecate
  typedef std::vector<l1extra::L1MuonParticleRef> VRl1muon;      //deprecate
  typedef std::vector<l1extra::L1JetParticleRef> VRl1jet;        //deprecate
  typedef std::vector<l1extra::L1EtMissParticleRef> VRl1etmiss;  //deprecate
  typedef std::vector<l1extra::L1HFRingsRef> VRl1hfrings;        //deprecate

  typedef l1t::MuonVectorRef VRl1tmuon;
  typedef l1t::MuonShowerVectorRef VRl1tmuonShower;
  typedef l1t::EGammaVectorRef VRl1tegamma;
  typedef l1t::JetVectorRef VRl1tjet;
  typedef l1t::TauVectorRef VRl1ttau;
  typedef l1t::EtSumVectorRef VRl1tetsum;

  /* Phase-2 */
  // This is a std::vector<TrackerMuonRef>,
  // and should be called TrackerMuonVectorRef upstream.
  // The L1T group should be made aware of that
  typedef l1t::TrackerMuonRefVector VRl1ttkmuon;
  typedef l1t::TkElectronVectorRef VRl1ttkele;
  typedef l1t::TkEmVectorRef VRl1ttkem;
  typedef l1t::PFJetVectorRef VRl1tpfjet;
  typedef l1t::HPSPFTauVectorRef VRl1thpspftau;
  typedef l1t::PFTauVectorRef VRl1tpftau;
  typedef l1t::PFTrackVectorRef VRl1tpftrack;
  typedef l1t::EtSumVectorP2Ref VRl1tp2etsum;

  typedef std::vector<reco::PFJetRef> VRpfjet;
  typedef std::vector<reco::PFTauRef> VRpftau;
  typedef std::vector<reco::PFMETRef> VRpfmet;

  class TriggerRefsCollections {
    /// data members
  private:
    /// physics type ids and Refs
    Vids photonIds_;
    VRphoton photonRefs_;
    Vids electronIds_;
    VRelectron electronRefs_;
    Vids muonIds_;
    VRmuon muonRefs_;
    Vids jetIds_;
    VRjet jetRefs_;
    Vids compositeIds_;
    VRcomposite compositeRefs_;
    Vids basemetIds_;
    VRbasemet basemetRefs_;
    Vids calometIds_;
    VRcalomet calometRefs_;
    Vids pixtrackIds_;
    VRpixtrack pixtrackRefs_;

    Vids l1emIds_;
    VRl1em l1emRefs_;
    Vids l1muonIds_;
    VRl1muon l1muonRefs_;
    Vids l1jetIds_;
    VRl1jet l1jetRefs_;
    Vids l1etmissIds_;
    VRl1etmiss l1etmissRefs_;
    Vids l1hfringsIds_;
    VRl1hfrings l1hfringsRefs_;

    Vids l1tmuonIds_;
    VRl1tmuon l1tmuonRefs_;
    Vids l1tmuonShowerIds_;
    VRl1tmuonShower l1tmuonShowerRefs_;
    Vids l1tegammaIds_;
    VRl1tegamma l1tegammaRefs_;
    Vids l1tjetIds_;
    VRl1tjet l1tjetRefs_;
    Vids l1ttauIds_;
    VRl1ttau l1ttauRefs_;
    Vids l1tetsumIds_;
    VRl1tetsum l1tetsumRefs_;

    /* Phase-2 */
    Vids l1ttkmuonIds_;
    VRl1ttkmuon l1ttkmuonRefs_;
    Vids l1ttkeleIds_;
    VRl1ttkele l1ttkeleRefs_;
    Vids l1ttkemIds_;
    VRl1ttkem l1ttkemRefs_;
    Vids l1tpfjetIds_;
    VRl1tpfjet l1tpfjetRefs_;
    Vids l1tpftauIds_;
    VRl1tpftau l1tpftauRefs_;
    Vids l1thpspftauIds_;
    VRl1thpspftau l1thpspftauRefs_;
    Vids l1tpftrackIds_;
    VRl1tpftrack l1tpftrackRefs_;
    Vids l1tp2etsumIds_;
    VRl1tp2etsum l1tp2etsumRefs_;

    Vids pfjetIds_;
    VRpfjet pfjetRefs_;
    Vids pftauIds_;
    VRpftau pftauRefs_;
    Vids pfmetIds_;
    VRpfmet pfmetRefs_;

    /// methods
  public:
    /// constructors
    TriggerRefsCollections()
        : photonIds_(),
          photonRefs_(),
          electronIds_(),
          electronRefs_(),
          muonIds_(),
          muonRefs_(),
          jetIds_(),
          jetRefs_(),
          compositeIds_(),
          compositeRefs_(),
          basemetIds_(),
          basemetRefs_(),
          calometIds_(),
          calometRefs_(),
          pixtrackIds_(),
          pixtrackRefs_(),

          l1emIds_(),
          l1emRefs_(),
          l1muonIds_(),
          l1muonRefs_(),
          l1jetIds_(),
          l1jetRefs_(),
          l1etmissIds_(),
          l1etmissRefs_(),
          l1hfringsIds_(),
          l1hfringsRefs_(),

          l1tmuonIds_(),
          l1tmuonRefs_(),
          l1tmuonShowerIds_(),
          l1tmuonShowerRefs_(),
          l1tegammaIds_(),
          l1tegammaRefs_(),
          l1tjetIds_(),
          l1tjetRefs_(),
          l1ttauIds_(),
          l1ttauRefs_(),
          l1tetsumIds_(),
          l1tetsumRefs_(),

          /* Phase-2 */
          l1ttkmuonIds_(),
          l1ttkmuonRefs_(),
          l1ttkeleIds_(),
          l1ttkeleRefs_(),
          l1ttkemIds_(),
          l1ttkemRefs_(),
          l1tpfjetIds_(),
          l1tpfjetRefs_(),
          l1tpftauIds_(),
          l1tpftauRefs_(),
          l1thpspftauIds_(),
          l1thpspftauRefs_(),
          l1tpftrackIds_(),
          l1tpftrackRefs_(),
          l1tp2etsumIds_(),
          l1tp2etsumRefs_(),

          pfjetIds_(),
          pfjetRefs_(),
          pftauIds_(),
          pftauRefs_(),
          pfmetIds_(),
          pfmetRefs_() {}

    /// utility
    void swap(TriggerRefsCollections& other) {
      std::swap(photonIds_, other.photonIds_);
      std::swap(photonRefs_, other.photonRefs_);
      std::swap(electronIds_, other.electronIds_);
      std::swap(electronRefs_, other.electronRefs_);
      std::swap(muonIds_, other.muonIds_);
      std::swap(muonRefs_, other.muonRefs_);
      std::swap(jetIds_, other.jetIds_);
      std::swap(jetRefs_, other.jetRefs_);
      std::swap(compositeIds_, other.compositeIds_);
      std::swap(compositeRefs_, other.compositeRefs_);
      std::swap(basemetIds_, other.basemetIds_);
      std::swap(basemetRefs_, other.basemetRefs_);
      std::swap(calometIds_, other.calometIds_);
      std::swap(calometRefs_, other.calometRefs_);
      std::swap(pixtrackIds_, other.pixtrackIds_);
      std::swap(pixtrackRefs_, other.pixtrackRefs_);

      std::swap(l1emIds_, other.l1emIds_);
      std::swap(l1emRefs_, other.l1emRefs_);
      std::swap(l1muonIds_, other.l1muonIds_);
      std::swap(l1muonRefs_, other.l1muonRefs_);
      std::swap(l1jetIds_, other.l1jetIds_);
      std::swap(l1jetRefs_, other.l1jetRefs_);
      std::swap(l1etmissIds_, other.l1etmissIds_);
      std::swap(l1etmissRefs_, other.l1etmissRefs_);
      std::swap(l1hfringsIds_, other.l1hfringsIds_);
      std::swap(l1hfringsRefs_, other.l1hfringsRefs_);

      std::swap(l1tmuonIds_, other.l1tmuonIds_);
      std::swap(l1tmuonRefs_, other.l1tmuonRefs_);
      std::swap(l1tmuonShowerIds_, other.l1tmuonShowerIds_);
      std::swap(l1tmuonShowerRefs_, other.l1tmuonShowerRefs_);
      std::swap(l1tegammaIds_, other.l1tegammaIds_);
      std::swap(l1tegammaRefs_, other.l1tegammaRefs_);
      std::swap(l1tjetIds_, other.l1tjetIds_);
      std::swap(l1tjetRefs_, other.l1tjetRefs_);
      std::swap(l1ttauIds_, other.l1ttauIds_);
      std::swap(l1ttauRefs_, other.l1ttauRefs_);
      std::swap(l1tetsumIds_, other.l1tetsumIds_);
      std::swap(l1tetsumRefs_, other.l1tetsumRefs_);

      /* Phase-2 */
      std::swap(l1ttkmuonIds_, other.l1ttkmuonIds_);
      std::swap(l1ttkmuonRefs_, other.l1ttkmuonRefs_);
      std::swap(l1ttkeleIds_, other.l1ttkeleIds_);
      std::swap(l1ttkeleRefs_, other.l1ttkeleRefs_);
      std::swap(l1ttkemIds_, other.l1ttkemIds_);
      std::swap(l1ttkemRefs_, other.l1ttkemRefs_);
      std::swap(l1tpfjetIds_, other.l1tpfjetIds_);
      std::swap(l1tpfjetRefs_, other.l1tpfjetRefs_);
      std::swap(l1tpftauIds_, other.l1tpftauIds_);
      std::swap(l1tpftauRefs_, other.l1tpftauRefs_);
      std::swap(l1thpspftauIds_, other.l1thpspftauIds_);
      std::swap(l1thpspftauRefs_, other.l1thpspftauRefs_);
      std::swap(l1tpftrackIds_, other.l1tpftrackIds_);
      std::swap(l1tpftrackRefs_, other.l1tpftrackRefs_);
      std::swap(l1tp2etsumIds_, other.l1tp2etsumIds_);
      std::swap(l1tp2etsumRefs_, other.l1tp2etsumRefs_);

      std::swap(pfjetIds_, other.pfjetIds_);
      std::swap(pfjetRefs_, other.pfjetRefs_);
      std::swap(pftauIds_, other.pftauIds_);
      std::swap(pftauRefs_, other.pftauRefs_);
      std::swap(pfmetIds_, other.pfmetIds_);
      std::swap(pfmetRefs_, other.pfmetRefs_);
    }

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
    void addObject(int id, const reco::METRef& ref) {
      basemetIds_.push_back(id);
      basemetRefs_.push_back(ref);
    }
    void addObject(int id, const reco::CaloMETRef& ref) {
      calometIds_.push_back(id);
      calometRefs_.push_back(ref);
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
    void addObject(int id, const l1extra::L1HFRingsRef& ref) {
      l1hfringsIds_.push_back(id);
      l1hfringsRefs_.push_back(ref);
    }
    void addObject(int id, const l1t::MuonRef& ref) {
      l1tmuonIds_.push_back(id);
      l1tmuonRefs_.push_back(ref);
    }
    void addObject(int id, const l1t::MuonShowerRef& ref) {
      l1tmuonShowerIds_.push_back(id);
      l1tmuonShowerRefs_.push_back(ref);
    }
    void addObject(int id, const l1t::EGammaRef& ref) {
      l1tegammaIds_.push_back(id);
      l1tegammaRefs_.push_back(ref);
    }
    void addObject(int id, const l1t::JetRef& ref) {
      l1tjetIds_.push_back(id);
      l1tjetRefs_.push_back(ref);
    }
    void addObject(int id, const l1t::TauRef& ref) {
      l1ttauIds_.push_back(id);
      l1ttauRefs_.push_back(ref);
    }
    void addObject(int id, const l1t::EtSumRef& ref) {
      l1tetsumIds_.push_back(id);
      l1tetsumRefs_.push_back(ref);
    }

    /* Phase-2 */
    void addObject(int id, const l1t::TrackerMuonRef& ref) {
      l1ttkmuonIds_.push_back(id);
      l1ttkmuonRefs_.push_back(ref);
    }
    void addObject(int id, const l1t::TkElectronRef& ref) {
      l1ttkeleIds_.push_back(id);
      l1ttkeleRefs_.push_back(ref);
    }
    void addObject(int id, const l1t::TkEmRef& ref) {
      l1ttkemIds_.push_back(id);
      l1ttkemRefs_.push_back(ref);
    }
    void addObject(int id, const l1t::PFJetRef& ref) {
      l1tpfjetIds_.push_back(id);
      l1tpfjetRefs_.push_back(ref);
    }
    void addObject(int id, const l1t::PFTauRef& ref) {
      l1tpftauIds_.push_back(id);
      l1tpftauRefs_.push_back(ref);
    }
    void addObject(int id, const l1t::HPSPFTauRef& ref) {
      l1thpspftauIds_.push_back(id);
      l1thpspftauRefs_.push_back(ref);
    }
    void addObject(int id, const l1t::PFTrackRef& ref) {
      l1tpftrackIds_.push_back(id);
      l1tpftrackRefs_.push_back(ref);
    }
    void addObject(int id, const l1t::EtSumP2Ref& ref) {
      l1tp2etsumIds_.push_back(id);
      l1tp2etsumRefs_.push_back(ref);
    }
    void addObject(int id, const reco::PFJetRef& ref) {
      pfjetIds_.push_back(id);
      pfjetRefs_.push_back(ref);
    }
    void addObject(int id, const reco::PFTauRef& ref) {
      pftauIds_.push_back(id);
      pftauRefs_.push_back(ref);
    }
    void addObject(int id, const reco::PFMETRef& ref) {
      pfmetIds_.push_back(id);
      pfmetRefs_.push_back(ref);
    }

    ///
    size_type addObjects(const Vids& ids, const VRphoton& refs) {
      assert(ids.size() == refs.size());
      photonIds_.insert(photonIds_.end(), ids.begin(), ids.end());
      photonRefs_.insert(photonRefs_.end(), refs.begin(), refs.end());
      return photonIds_.size();
    }
    size_type addObjects(const Vids& ids, const VRelectron& refs) {
      assert(ids.size() == refs.size());
      electronIds_.insert(electronIds_.end(), ids.begin(), ids.end());
      electronRefs_.insert(electronRefs_.end(), refs.begin(), refs.end());
      return electronIds_.size();
    }
    size_type addObjects(const Vids& ids, const VRmuon& refs) {
      assert(ids.size() == refs.size());
      muonIds_.insert(muonIds_.end(), ids.begin(), ids.end());
      muonRefs_.insert(muonRefs_.end(), refs.begin(), refs.end());
      return muonIds_.size();
    }
    size_type addObjects(const Vids& ids, const VRjet& refs) {
      assert(ids.size() == refs.size());
      jetIds_.insert(jetIds_.end(), ids.begin(), ids.end());
      jetRefs_.insert(jetRefs_.end(), refs.begin(), refs.end());
      return jetIds_.size();
    }
    size_type addObjects(const Vids& ids, const VRcomposite& refs) {
      assert(ids.size() == refs.size());
      compositeIds_.insert(compositeIds_.end(), ids.begin(), ids.end());
      compositeRefs_.insert(compositeRefs_.end(), refs.begin(), refs.end());
      return compositeIds_.size();
    }
    size_type addObjects(const Vids& ids, const VRbasemet& refs) {
      assert(ids.size() == refs.size());
      basemetIds_.insert(basemetIds_.end(), ids.begin(), ids.end());
      basemetRefs_.insert(basemetRefs_.end(), refs.begin(), refs.end());
      return basemetIds_.size();
    }
    size_type addObjects(const Vids& ids, const VRcalomet& refs) {
      assert(ids.size() == refs.size());
      calometIds_.insert(calometIds_.end(), ids.begin(), ids.end());
      calometRefs_.insert(calometRefs_.end(), refs.begin(), refs.end());
      return calometIds_.size();
    }
    size_type addObjects(const Vids& ids, const VRpixtrack& refs) {
      assert(ids.size() == refs.size());
      pixtrackIds_.insert(pixtrackIds_.end(), ids.begin(), ids.end());
      pixtrackRefs_.insert(pixtrackRefs_.end(), refs.begin(), refs.end());
      return pixtrackIds_.size();
    }

    size_type addObjects(const Vids& ids, const VRl1em& refs) {
      assert(ids.size() == refs.size());
      l1emIds_.insert(l1emIds_.end(), ids.begin(), ids.end());
      l1emRefs_.insert(l1emRefs_.end(), refs.begin(), refs.end());
      return l1emIds_.size();
    }
    size_type addObjects(const Vids& ids, const VRl1muon& refs) {
      assert(ids.size() == refs.size());
      l1muonIds_.insert(l1muonIds_.end(), ids.begin(), ids.end());
      l1muonRefs_.insert(l1muonRefs_.end(), refs.begin(), refs.end());
      return l1muonIds_.size();
    }
    size_type addObjects(const Vids& ids, const VRl1jet& refs) {
      assert(ids.size() == refs.size());
      l1jetIds_.insert(l1jetIds_.end(), ids.begin(), ids.end());
      l1jetRefs_.insert(l1jetRefs_.end(), refs.begin(), refs.end());
      return l1jetIds_.size();
    }
    size_type addObjects(const Vids& ids, const VRl1etmiss& refs) {
      assert(ids.size() == refs.size());
      l1etmissIds_.insert(l1etmissIds_.end(), ids.begin(), ids.end());
      l1etmissRefs_.insert(l1etmissRefs_.end(), refs.begin(), refs.end());
      return l1etmissIds_.size();
    }
    size_type addObjects(const Vids& ids, const VRl1tmuon& refs) {
      assert(ids.size() == refs.size());
      l1tmuonIds_.insert(l1tmuonIds_.end(), ids.begin(), ids.end());
      l1tmuonRefs_.insert(l1tmuonRefs_.end(), refs.begin(), refs.end());
      return l1tmuonIds_.size();
    }
    size_type addObjects(const Vids& ids, const VRl1tmuonShower& refs) {
      assert(ids.size() == refs.size());
      l1tmuonShowerIds_.insert(l1tmuonShowerIds_.end(), ids.begin(), ids.end());
      l1tmuonShowerRefs_.insert(l1tmuonShowerRefs_.end(), refs.begin(), refs.end());
      return l1tmuonShowerIds_.size();
    }
    size_type addObjects(const Vids& ids, const VRl1tegamma& refs) {
      assert(ids.size() == refs.size());
      l1tegammaIds_.insert(l1tegammaIds_.end(), ids.begin(), ids.end());
      l1tegammaRefs_.insert(l1tegammaRefs_.end(), refs.begin(), refs.end());
      return l1tegammaIds_.size();
    }
    size_type addObjects(const Vids& ids, const VRl1tjet& refs) {
      assert(ids.size() == refs.size());
      l1tjetIds_.insert(l1tjetIds_.end(), ids.begin(), ids.end());
      l1tjetRefs_.insert(l1tjetRefs_.end(), refs.begin(), refs.end());
      return l1tjetIds_.size();
    }
    size_type addObjects(const Vids& ids, const VRl1ttau& refs) {
      assert(ids.size() == refs.size());
      l1ttauIds_.insert(l1ttauIds_.end(), ids.begin(), ids.end());
      l1ttauRefs_.insert(l1ttauRefs_.end(), refs.begin(), refs.end());
      return l1ttauIds_.size();
    }
    size_type addObjects(const Vids& ids, const VRl1tetsum& refs) {
      assert(ids.size() == refs.size());
      l1tetsumIds_.insert(l1tetsumIds_.end(), ids.begin(), ids.end());
      l1tetsumRefs_.insert(l1tetsumRefs_.end(), refs.begin(), refs.end());
      return l1tetsumIds_.size();
    }
    size_type addObjects(const Vids& ids, const VRl1hfrings& refs) {
      assert(ids.size() == refs.size());
      l1hfringsIds_.insert(l1hfringsIds_.end(), ids.begin(), ids.end());
      l1hfringsRefs_.insert(l1hfringsRefs_.end(), refs.begin(), refs.end());
      return l1hfringsIds_.size();
    }

    /* Phase-2 */
    size_type addObjects(const Vids& ids, const VRl1ttkmuon& refs) {
      assert(ids.size() == refs.size());
      l1ttkmuonIds_.insert(l1ttkmuonIds_.end(), ids.begin(), ids.end());
      l1ttkmuonRefs_.insert(l1ttkmuonRefs_.end(), refs.begin(), refs.end());
      return l1ttkmuonIds_.size();
    }
    size_type addObjects(const Vids& ids, const VRl1ttkele& refs) {
      assert(ids.size() == refs.size());
      l1ttkeleIds_.insert(l1ttkeleIds_.end(), ids.begin(), ids.end());
      l1ttkeleRefs_.insert(l1ttkeleRefs_.end(), refs.begin(), refs.end());
      return l1ttkeleIds_.size();
    }
    size_type addObjects(const Vids& ids, const VRl1ttkem& refs) {
      assert(ids.size() == refs.size());
      l1ttkemIds_.insert(l1ttkemIds_.end(), ids.begin(), ids.end());
      l1ttkemRefs_.insert(l1ttkemRefs_.end(), refs.begin(), refs.end());
      return l1ttkemIds_.size();
    }
    size_type addObjects(const Vids& ids, const VRl1tpfjet& refs) {
      assert(ids.size() == refs.size());
      l1tpfjetIds_.insert(l1tpfjetIds_.end(), ids.begin(), ids.end());
      l1tpfjetRefs_.insert(l1tpfjetRefs_.end(), refs.begin(), refs.end());
      return l1tpfjetIds_.size();
    }
    size_type addObjects(const Vids& ids, const VRl1tpftau& refs) {
      assert(ids.size() == refs.size());
      l1tpftauIds_.insert(l1tpftauIds_.end(), ids.begin(), ids.end());
      l1tpftauRefs_.insert(l1tpftauRefs_.end(), refs.begin(), refs.end());
      return l1tpftauIds_.size();
    }
    size_type addObjects(const Vids& ids, const VRl1thpspftau& refs) {
      assert(ids.size() == refs.size());
      l1thpspftauIds_.insert(l1thpspftauIds_.end(), ids.begin(), ids.end());
      l1thpspftauRefs_.insert(l1thpspftauRefs_.end(), refs.begin(), refs.end());
      return l1thpspftauIds_.size();
    }
    size_type addObjects(const Vids& ids, const VRl1tpftrack& refs) {
      assert(ids.size() == refs.size());
      l1tpftrackIds_.insert(l1tpftrackIds_.end(), ids.begin(), ids.end());
      l1tpftrackRefs_.insert(l1tpftrackRefs_.end(), refs.begin(), refs.end());
      return l1tpftrackIds_.size();
    }
    size_type addObjects(const Vids& ids, const VRl1tp2etsum& refs) {
      assert(ids.size() == refs.size());
      l1tp2etsumIds_.insert(l1tp2etsumIds_.end(), ids.begin(), ids.end());
      l1tp2etsumRefs_.insert(l1tp2etsumRefs_.end(), refs.begin(), refs.end());
      return l1tp2etsumIds_.size();
    }

    size_type addObjects(const Vids& ids, const VRpfjet& refs) {
      assert(ids.size() == refs.size());
      pfjetIds_.insert(pfjetIds_.end(), ids.begin(), ids.end());
      pfjetRefs_.insert(pfjetRefs_.end(), refs.begin(), refs.end());
      return pfjetIds_.size();
    }
    size_type addObjects(const Vids& ids, const VRpftau& refs) {
      assert(ids.size() == refs.size());
      pftauIds_.insert(pftauIds_.end(), ids.begin(), ids.end());
      pftauRefs_.insert(pftauRefs_.end(), refs.begin(), refs.end());
      return pftauIds_.size();
    }
    size_type addObjects(const Vids& ids, const VRpfmet& refs) {
      assert(ids.size() == refs.size());
      pfmetIds_.insert(pfmetIds_.end(), ids.begin(), ids.end());
      pfmetRefs_.insert(pfmetRefs_.end(), refs.begin(), refs.end());
      return pfmetIds_.size();
    }

    /// various physics-level getters:
    void getObjects(Vids& ids, VRphoton& refs) const { getObjects(ids, refs, 0, photonIds_.size()); }
    void getObjects(Vids& ids, VRphoton& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= photonIds_.size());
      const size_type n(end - begin);
      ids.resize(n);
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        ids[j] = photonIds_[i];
        refs[j] = photonRefs_[i];
        ++j;
      }
    }
    void getObjects(int id, VRphoton& refs) const { getObjects(id, refs, 0, photonIds_.size()); }
    void getObjects(int id, VRphoton& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= photonIds_.size());
      size_type n(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == photonIds_[i]) {
          ++n;
        }
      }
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == photonIds_[i]) {
          refs[j] = photonRefs_[i];
          ++j;
        }
      }
      return;
    }

    void getObjects(Vids& ids, VRelectron& refs) const { getObjects(ids, refs, 0, electronIds_.size()); }
    void getObjects(Vids& ids, VRelectron& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= electronIds_.size());
      const size_type n(end - begin);
      ids.resize(n);
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        ids[j] = electronIds_[i];
        refs[j] = electronRefs_[i];
        ++j;
      }
    }
    void getObjects(int id, VRelectron& refs) const { getObjects(id, refs, 0, electronIds_.size()); }
    void getObjects(int id, VRelectron& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= electronIds_.size());
      size_type n(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == electronIds_[i]) {
          ++n;
        }
      }
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == electronIds_[i]) {
          refs[j] = electronRefs_[i];
          ++j;
        }
      }
      return;
    }

    void getObjects(Vids& ids, VRmuon& refs) const { getObjects(ids, refs, 0, muonIds_.size()); }
    void getObjects(Vids& ids, VRmuon& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= muonIds_.size());
      const size_type n(end - begin);
      ids.resize(n);
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        ids[j] = muonIds_[i];
        refs[j] = muonRefs_[i];
        ++j;
      }
    }
    void getObjects(int id, VRmuon& refs) const { getObjects(id, refs, 0, muonIds_.size()); }
    void getObjects(int id, VRmuon& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= muonIds_.size());
      size_type n(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == muonIds_[i]) {
          ++n;
        }
      }
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == muonIds_[i]) {
          refs[j] = muonRefs_[i];
          ++j;
        }
      }
      return;
    }

    void getObjects(Vids& ids, VRjet& refs) const { getObjects(ids, refs, 0, jetIds_.size()); }
    void getObjects(Vids& ids, VRjet& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= jetIds_.size());
      const size_type n(end - begin);
      ids.resize(n);
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        ids[j] = jetIds_[i];
        refs[j] = jetRefs_[i];
        ++j;
      }
    }
    void getObjects(int id, VRjet& refs) const { getObjects(id, refs, 0, jetIds_.size()); }
    void getObjects(int id, VRjet& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= jetIds_.size());
      size_type n(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == jetIds_[i]) {
          ++n;
        }
      }
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == jetIds_[i]) {
          refs[j] = jetRefs_[i];
          ++j;
        }
      }
      return;
    }

    void getObjects(Vids& ids, VRcomposite& refs) const { getObjects(ids, refs, 0, compositeIds_.size()); }
    void getObjects(Vids& ids, VRcomposite& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= compositeIds_.size());
      const size_type n(end - begin);
      ids.resize(n);
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        ids[j] = compositeIds_[i];
        refs[j] = compositeRefs_[i];
        ++j;
      }
    }
    void getObjects(int id, VRcomposite& refs) const { getObjects(id, refs, 0, compositeIds_.size()); }
    void getObjects(int id, VRcomposite& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= compositeIds_.size());
      size_type n(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == compositeIds_[i]) {
          ++n;
        }
      }
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == compositeIds_[i]) {
          refs[j] = compositeRefs_[i];
          ++j;
        }
      }
      return;
    }

    void getObjects(Vids& ids, VRbasemet& refs) const { getObjects(ids, refs, 0, basemetIds_.size()); }
    void getObjects(Vids& ids, VRbasemet& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= basemetIds_.size());
      const size_type n(end - begin);
      ids.resize(n);
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        ids[j] = basemetIds_[i];
        refs[j] = basemetRefs_[i];
        ++j;
      }
    }
    void getObjects(int id, VRbasemet& refs) const { getObjects(id, refs, 0, basemetIds_.size()); }
    void getObjects(int id, VRbasemet& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= basemetIds_.size());
      size_type n(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == basemetIds_[i]) {
          ++n;
        }
      }
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == basemetIds_[i]) {
          refs[j] = basemetRefs_[i];
          ++j;
        }
      }
      return;
    }

    void getObjects(Vids& ids, VRcalomet& refs) const { getObjects(ids, refs, 0, calometIds_.size()); }
    void getObjects(Vids& ids, VRcalomet& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= calometIds_.size());
      const size_type n(end - begin);
      ids.resize(n);
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        ids[j] = calometIds_[i];
        refs[j] = calometRefs_[i];
        ++j;
      }
    }
    void getObjects(int id, VRcalomet& refs) const { getObjects(id, refs, 0, calometIds_.size()); }
    void getObjects(int id, VRcalomet& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= calometIds_.size());
      size_type n(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == calometIds_[i]) {
          ++n;
        }
      }
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == calometIds_[i]) {
          refs[j] = calometRefs_[i];
          ++j;
        }
      }
      return;
    }

    void getObjects(Vids& ids, VRpixtrack& refs) const { getObjects(ids, refs, 0, pixtrackIds_.size()); }
    void getObjects(Vids& ids, VRpixtrack& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= pixtrackIds_.size());
      const size_type n(end - begin);
      ids.resize(n);
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        ids[j] = pixtrackIds_[i];
        refs[j] = pixtrackRefs_[i];
        ++j;
      }
    }
    void getObjects(int id, VRpixtrack& refs) const { getObjects(id, refs, 0, pixtrackIds_.size()); }
    void getObjects(int id, VRpixtrack& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= pixtrackIds_.size());
      size_type n(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == pixtrackIds_[i]) {
          ++n;
        }
      }
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == pixtrackIds_[i]) {
          refs[j] = pixtrackRefs_[i];
          ++j;
        }
      }
      return;
    }

    void getObjects(Vids& ids, VRl1em& refs) const { getObjects(ids, refs, 0, l1emIds_.size()); }
    void getObjects(Vids& ids, VRl1em& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1emIds_.size());
      const size_type n(end - begin);
      ids.resize(n);
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        ids[j] = l1emIds_[i];
        refs[j] = l1emRefs_[i];
        ++j;
      }
    }
    void getObjects(int id, VRl1em& refs) const { getObjects(id, refs, 0, l1emIds_.size()); }
    void getObjects(int id, VRl1em& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1emIds_.size());
      size_type n(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1emIds_[i]) {
          ++n;
        }
      }
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1emIds_[i]) {
          refs[j] = l1emRefs_[i];
          ++j;
        }
      }
      return;
    }

    void getObjects(Vids& ids, VRl1muon& refs) const { getObjects(ids, refs, 0, l1muonIds_.size()); }
    void getObjects(Vids& ids, VRl1muon& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1muonIds_.size());
      const size_type n(end - begin);
      ids.resize(n);
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        ids[j] = l1muonIds_[i];
        refs[j] = l1muonRefs_[i];
        ++j;
      }
    }
    void getObjects(int id, VRl1muon& refs) const { getObjects(id, refs, 0, l1muonIds_.size()); }
    void getObjects(int id, VRl1muon& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1muonIds_.size());
      size_type n(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1muonIds_[i]) {
          ++n;
        }
      }
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1muonIds_[i]) {
          refs[j] = l1muonRefs_[i];
          ++j;
        }
      }
      return;
    }

    void getObjects(Vids& ids, VRl1jet& refs) const { getObjects(ids, refs, 0, l1jetIds_.size()); }
    void getObjects(Vids& ids, VRl1jet& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1jetIds_.size());
      const size_type n(end - begin);
      ids.resize(n);
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        ids[j] = l1jetIds_[i];
        refs[j] = l1jetRefs_[i];
        ++j;
      }
    }
    void getObjects(int id, VRl1jet& refs) const { getObjects(id, refs, 0, l1jetIds_.size()); }
    void getObjects(int id, VRl1jet& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1jetIds_.size());
      size_type n(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1jetIds_[i]) {
          ++n;
        }
      }
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1jetIds_[i]) {
          refs[j] = l1jetRefs_[i];
          ++j;
        }
      }
      return;
    }

    void getObjects(Vids& ids, VRl1etmiss& refs) const { getObjects(ids, refs, 0, l1etmissIds_.size()); }
    void getObjects(Vids& ids, VRl1etmiss& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1etmissIds_.size());
      const size_type n(end - begin);
      ids.resize(n);
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        ids[j] = l1etmissIds_[i];
        refs[j] = l1etmissRefs_[i];
        ++j;
      }
    }
    void getObjects(int id, VRl1etmiss& refs) const { getObjects(id, refs, 0, l1etmissIds_.size()); }
    void getObjects(int id, VRl1etmiss& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1etmissIds_.size());
      size_type n(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1etmissIds_[i]) {
          ++n;
        }
      }
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1etmissIds_[i]) {
          refs[j] = l1etmissRefs_[i];
          ++j;
        }
      }
      return;
    }

    void getObjects(Vids& ids, VRl1hfrings& refs) const { getObjects(ids, refs, 0, l1hfringsIds_.size()); }
    void getObjects(Vids& ids, VRl1hfrings& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1hfringsIds_.size());
      const size_type n(end - begin);
      ids.resize(n);
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        ids[j] = l1hfringsIds_[i];
        refs[j] = l1hfringsRefs_[i];
        ++j;
      }
    }
    void getObjects(int id, VRl1hfrings& refs) const { getObjects(id, refs, 0, l1hfringsIds_.size()); }
    void getObjects(int id, VRl1hfrings& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1hfringsIds_.size());
      size_type n(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1hfringsIds_[i]) {
          ++n;
        }
      }
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1hfringsIds_[i]) {
          refs[j] = l1hfringsRefs_[i];
          ++j;
        }
      }
      return;
    }

    void getObjects(Vids& ids, VRl1tmuon& refs) const { getObjects(ids, refs, 0, l1tmuonIds_.size()); }
    void getObjects(Vids& ids, VRl1tmuon& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1tmuonIds_.size());
      const size_type n(end - begin);
      ids.resize(n);
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        ids[j] = l1tmuonIds_[i];
        refs[j] = l1tmuonRefs_[i];
        ++j;
      }
    }
    void getObjects(int id, VRl1tmuon& refs) const { getObjects(id, refs, 0, l1tmuonIds_.size()); }
    void getObjects(int id, VRl1tmuon& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1tmuonIds_.size());
      size_type n(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1tmuonIds_[i]) {
          ++n;
        }
      }
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1tmuonIds_[i]) {
          refs[j] = l1tmuonRefs_[i];
          ++j;
        }
      }
      return;
    }

    void getObjects(Vids& ids, VRl1tmuonShower& refs) const { getObjects(ids, refs, 0, l1tmuonShowerIds_.size()); }
    void getObjects(Vids& ids, VRl1tmuonShower& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1tmuonShowerIds_.size());
      const size_type n(end - begin);
      ids.resize(n);
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        ids[j] = l1tmuonShowerIds_[i];
        refs[j] = l1tmuonShowerRefs_[i];
        ++j;
      }
    }
    void getObjects(int id, VRl1tmuonShower& refs) const { getObjects(id, refs, 0, l1tmuonShowerIds_.size()); }
    void getObjects(int id, VRl1tmuonShower& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1tmuonShowerIds_.size());
      size_type n(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1tmuonShowerIds_[i]) {
          ++n;
        }
      }
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1tmuonShowerIds_[i]) {
          refs[j] = l1tmuonShowerRefs_[i];
          ++j;
        }
      }
      return;
    }

    void getObjects(Vids& ids, VRl1tegamma& refs) const { getObjects(ids, refs, 0, l1tegammaIds_.size()); }
    void getObjects(Vids& ids, VRl1tegamma& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1tegammaIds_.size());
      const size_type n(end - begin);
      ids.resize(n);
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        ids[j] = l1tegammaIds_[i];
        refs[j] = l1tegammaRefs_[i];
        ++j;
      }
    }
    void getObjects(int id, VRl1tegamma& refs) const { getObjects(id, refs, 0, l1tegammaIds_.size()); }
    void getObjects(int id, VRl1tegamma& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1tegammaIds_.size());
      size_type n(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1tegammaIds_[i]) {
          ++n;
        }
      }
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1tegammaIds_[i]) {
          refs[j] = l1tegammaRefs_[i];
          ++j;
        }
      }
      return;
    }

    void getObjects(Vids& ids, VRl1tjet& refs) const { getObjects(ids, refs, 0, l1tjetIds_.size()); }
    void getObjects(Vids& ids, VRl1tjet& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1tjetIds_.size());
      const size_type n(end - begin);
      ids.resize(n);
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        ids[j] = l1tjetIds_[i];
        refs[j] = l1tjetRefs_[i];
        ++j;
      }
    }
    void getObjects(int id, VRl1tjet& refs) const { getObjects(id, refs, 0, l1tjetIds_.size()); }
    void getObjects(int id, VRl1tjet& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1tjetIds_.size());
      size_type n(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1tjetIds_[i]) {
          ++n;
        }
      }
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1tjetIds_[i]) {
          refs[j] = l1tjetRefs_[i];
          ++j;
        }
      }
      return;
    }

    void getObjects(Vids& ids, VRl1ttau& refs) const { getObjects(ids, refs, 0, l1ttauIds_.size()); }
    void getObjects(Vids& ids, VRl1ttau& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1ttauIds_.size());
      const size_type n(end - begin);
      ids.resize(n);
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        ids[j] = l1ttauIds_[i];
        refs[j] = l1ttauRefs_[i];
        ++j;
      }
    }
    void getObjects(int id, VRl1ttau& refs) const { getObjects(id, refs, 0, l1ttauIds_.size()); }
    void getObjects(int id, VRl1ttau& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1ttauIds_.size());
      size_type n(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1ttauIds_[i]) {
          ++n;
        }
      }
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1ttauIds_[i]) {
          refs[j] = l1ttauRefs_[i];
          ++j;
        }
      }
      return;
    }

    void getObjects(Vids& ids, VRl1tetsum& refs) const { getObjects(ids, refs, 0, l1tetsumIds_.size()); }
    void getObjects(Vids& ids, VRl1tetsum& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1tetsumIds_.size());
      const size_type n(end - begin);
      ids.resize(n);
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        ids[j] = l1tetsumIds_[i];
        refs[j] = l1tetsumRefs_[i];
        ++j;
      }
    }
    void getObjects(int id, VRl1tetsum& refs) const { getObjects(id, refs, 0, l1tetsumIds_.size()); }
    void getObjects(int id, VRl1tetsum& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1tetsumIds_.size());
      size_type n(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1tetsumIds_[i]) {
          ++n;
        }
      }
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1tetsumIds_[i]) {
          refs[j] = l1tetsumRefs_[i];
          ++j;
        }
      }
      return;
    }

    /* Phase-2 */
    void getObjects(Vids& ids, VRl1ttkmuon& refs) const { getObjects(ids, refs, 0, l1ttkmuonIds_.size()); }
    void getObjects(Vids& ids, VRl1ttkmuon& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1ttkmuonIds_.size());
      const size_type n(end - begin);
      ids.resize(n);
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        ids[j] = l1ttkmuonIds_[i];
        refs[j] = l1ttkmuonRefs_[i];
        ++j;
      }
    }
    void getObjects(int id, VRl1ttkmuon& refs) const { getObjects(id, refs, 0, l1ttkmuonIds_.size()); }
    void getObjects(int id, VRl1ttkmuon& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1ttkmuonIds_.size());
      size_type n(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1ttkmuonIds_[i]) {
          ++n;
        }
      }
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1ttkmuonIds_[i]) {
          refs[j] = l1ttkmuonRefs_[i];
          ++j;
        }
      }
      return;
    }

    void getObjects(Vids& ids, VRl1ttkele& refs) const { getObjects(ids, refs, 0, l1ttkeleIds_.size()); }
    void getObjects(Vids& ids, VRl1ttkele& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1ttkeleIds_.size());
      const size_type n(end - begin);
      ids.resize(n);
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        ids[j] = l1ttkeleIds_[i];
        refs[j] = l1ttkeleRefs_[i];
        ++j;
      }
    }
    void getObjects(int id, VRl1ttkele& refs) const { getObjects(id, refs, 0, l1ttkeleIds_.size()); }
    void getObjects(int id, VRl1ttkele& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1ttkeleIds_.size());
      size_type n(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1ttkeleIds_[i]) {
          ++n;
        }
      }
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1ttkeleIds_[i]) {
          refs[j] = l1ttkeleRefs_[i];
          ++j;
        }
      }
      return;
    }

    void getObjects(Vids& ids, VRl1ttkem& refs) const { getObjects(ids, refs, 0, l1ttkemIds_.size()); }
    void getObjects(Vids& ids, VRl1ttkem& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1ttkemIds_.size());
      const size_type n(end - begin);
      ids.resize(n);
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        ids[j] = l1ttkemIds_[i];
        refs[j] = l1ttkemRefs_[i];
        ++j;
      }
    }
    void getObjects(int id, VRl1ttkem& refs) const { getObjects(id, refs, 0, l1ttkemIds_.size()); }
    void getObjects(int id, VRl1ttkem& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1ttkemIds_.size());
      size_type n(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1ttkemIds_[i]) {
          ++n;
        }
      }
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1ttkemIds_[i]) {
          refs[j] = l1ttkemRefs_[i];
          ++j;
        }
      }
      return;
    }

    void getObjects(Vids& ids, VRl1tpfjet& refs) const { getObjects(ids, refs, 0, l1tpfjetIds_.size()); }
    void getObjects(Vids& ids, VRl1tpfjet& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1tpfjetIds_.size());
      const size_type n(end - begin);
      ids.resize(n);
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        ids[j] = l1tpfjetIds_[i];
        refs[j] = l1tpfjetRefs_[i];
        ++j;
      }
    }
    void getObjects(int id, VRl1tpfjet& refs) const { getObjects(id, refs, 0, l1tpfjetIds_.size()); }
    void getObjects(int id, VRl1tpfjet& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1tpfjetIds_.size());
      size_type n(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1tpfjetIds_[i]) {
          ++n;
        }
      }
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1tpfjetIds_[i]) {
          refs[j] = l1tpfjetRefs_[i];
          ++j;
        }
      }
      return;
    }

    void getObjects(Vids& ids, VRl1tpftau& refs) const { getObjects(ids, refs, 0, l1tpftauIds_.size()); }
    void getObjects(Vids& ids, VRl1tpftau& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1tpftauIds_.size());
      const size_type n(end - begin);
      ids.resize(n);
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        ids[j] = l1tpftauIds_[i];
        refs[j] = l1tpftauRefs_[i];
        ++j;
      }
    }
    void getObjects(int id, VRl1tpftau& refs) const { getObjects(id, refs, 0, l1tpftauIds_.size()); }
    void getObjects(int id, VRl1tpftau& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1tpftauIds_.size());
      size_type n(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1tpftauIds_[i]) {
          ++n;
        }
      }
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1tpftauIds_[i]) {
          refs[j] = l1tpftauRefs_[i];
          ++j;
        }
      }
      return;
    }

    void getObjects(Vids& ids, VRl1thpspftau& refs) const { getObjects(ids, refs, 0, l1thpspftauIds_.size()); }
    void getObjects(Vids& ids, VRl1thpspftau& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1thpspftauIds_.size());
      const size_type n(end - begin);
      ids.resize(n);
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        ids[j] = l1thpspftauIds_[i];
        refs[j] = l1thpspftauRefs_[i];
        ++j;
      }
    }
    void getObjects(int id, VRl1thpspftau& refs) const { getObjects(id, refs, 0, l1thpspftauIds_.size()); }
    void getObjects(int id, VRl1thpspftau& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1thpspftauIds_.size());
      size_type n(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1thpspftauIds_[i]) {
          ++n;
        }
      }
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1thpspftauIds_[i]) {
          refs[j] = l1thpspftauRefs_[i];
          ++j;
        }
      }
      return;
    }

    void getObjects(Vids& ids, VRl1tpftrack& refs) const { getObjects(ids, refs, 0, l1tpftrackIds_.size()); }
    void getObjects(Vids& ids, VRl1tpftrack& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1tpftrackIds_.size());
      const size_type n(end - begin);
      ids.resize(n);
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        ids[j] = l1tpftrackIds_[i];
        refs[j] = l1tpftrackRefs_[i];
        ++j;
      }
    }
    void getObjects(int id, VRl1tpftrack& refs) const { getObjects(id, refs, 0, l1tpftrackIds_.size()); }
    void getObjects(int id, VRl1tpftrack& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1tpftrackIds_.size());
      size_type n(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1tpftrackIds_[i]) {
          ++n;
        }
      }
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1tpftrackIds_[i]) {
          refs[j] = l1tpftrackRefs_[i];
          ++j;
        }
      }
      return;
    }

    void getObjects(Vids& ids, VRl1tp2etsum& refs) const { getObjects(ids, refs, 0, l1tp2etsumIds_.size()); }
    void getObjects(Vids& ids, VRl1tp2etsum& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1tp2etsumIds_.size());
      const size_type n(end - begin);
      ids.resize(n);
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        ids[j] = l1tp2etsumIds_[i];
        refs[j] = l1tp2etsumRefs_[i];
        ++j;
      }
    }
    void getObjects(int id, VRl1tp2etsum& refs) const { getObjects(id, refs, 0, l1tp2etsumIds_.size()); }
    void getObjects(int id, VRl1tp2etsum& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= l1tp2etsumIds_.size());
      size_type n(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1tp2etsumIds_[i]) {
          ++n;
        }
      }
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == l1tp2etsumIds_[i]) {
          refs[j] = l1tp2etsumRefs_[i];
          ++j;
        }
      }
      return;
    }

    void getObjects(Vids& ids, VRpfjet& refs) const { getObjects(ids, refs, 0, pfjetIds_.size()); }
    void getObjects(Vids& ids, VRpfjet& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= pfjetIds_.size());
      const size_type n(end - begin);
      ids.resize(n);
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        ids[j] = pfjetIds_[i];
        refs[j] = pfjetRefs_[i];
        ++j;
      }
    }
    void getObjects(int id, VRpfjet& refs) const { getObjects(id, refs, 0, pfjetIds_.size()); }
    void getObjects(int id, VRpfjet& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= pfjetIds_.size());
      size_type n(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == pfjetIds_[i]) {
          ++n;
        }
      }
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == pfjetIds_[i]) {
          refs[j] = pfjetRefs_[i];
          ++j;
        }
      }
      return;
    }

    void getObjects(Vids& ids, VRpftau& refs) const { getObjects(ids, refs, 0, pftauIds_.size()); }
    void getObjects(Vids& ids, VRpftau& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= pftauIds_.size());
      const size_type n(end - begin);
      ids.resize(n);
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        ids[j] = pftauIds_[i];
        refs[j] = pftauRefs_[i];
        ++j;
      }
    }
    void getObjects(int id, VRpftau& refs) const { getObjects(id, refs, 0, pftauIds_.size()); }
    void getObjects(int id, VRpftau& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= pftauIds_.size());
      size_type n(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == pftauIds_[i]) {
          ++n;
        }
      }
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == pftauIds_[i]) {
          refs[j] = pftauRefs_[i];
          ++j;
        }
      }
      return;
    }

    void getObjects(Vids& ids, VRpfmet& refs) const { getObjects(ids, refs, 0, pfmetIds_.size()); }
    void getObjects(Vids& ids, VRpfmet& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= pfmetIds_.size());
      const size_type n(end - begin);
      ids.resize(n);
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        ids[j] = pfmetIds_[i];
        refs[j] = pfmetRefs_[i];
        ++j;
      }
    }
    void getObjects(int id, VRpfmet& refs) const { getObjects(id, refs, 0, pfmetIds_.size()); }
    void getObjects(int id, VRpfmet& refs, size_type begin, size_type end) const {
      assert(begin <= end);
      assert(end <= pfmetIds_.size());
      size_type n(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == pfmetIds_[i]) {
          ++n;
        }
      }
      refs.resize(n);
      size_type j(0);
      for (size_type i = begin; i != end; ++i) {
        if (id == pfmetIds_[i]) {
          refs[j] = pfmetRefs_[i];
          ++j;
        }
      }
      return;
    }

    /// low-level getters for data members
    size_type photonSize() const { return photonIds_.size(); }
    const Vids& photonIds() const { return photonIds_; }
    const VRphoton& photonRefs() const { return photonRefs_; }

    size_type electronSize() const { return electronIds_.size(); }
    const Vids& electronIds() const { return electronIds_; }
    const VRelectron& electronRefs() const { return electronRefs_; }

    size_type muonSize() const { return muonIds_.size(); }
    const Vids& muonIds() const { return muonIds_; }
    const VRmuon& muonRefs() const { return muonRefs_; }

    size_type jetSize() const { return jetIds_.size(); }
    const Vids& jetIds() const { return jetIds_; }
    const VRjet& jetRefs() const { return jetRefs_; }

    size_type compositeSize() const { return compositeIds_.size(); }
    const Vids& compositeIds() const { return compositeIds_; }
    const VRcomposite& compositeRefs() const { return compositeRefs_; }

    size_type basemetSize() const { return basemetIds_.size(); }
    const Vids& basemetIds() const { return basemetIds_; }
    const VRbasemet& basemetRefs() const { return basemetRefs_; }

    size_type calometSize() const { return calometIds_.size(); }
    const Vids& calometIds() const { return calometIds_; }
    const VRcalomet& calometRefs() const { return calometRefs_; }

    size_type pixtrackSize() const { return pixtrackIds_.size(); }
    const Vids& pixtrackIds() const { return pixtrackIds_; }
    const VRpixtrack& pixtrackRefs() const { return pixtrackRefs_; }

    size_type l1emSize() const { return l1emIds_.size(); }
    const Vids& l1emIds() const { return l1emIds_; }
    const VRl1em& l1emRefs() const { return l1emRefs_; }

    size_type l1muonSize() const { return l1muonIds_.size(); }
    const Vids& l1muonIds() const { return l1muonIds_; }
    const VRl1muon& l1muonRefs() const { return l1muonRefs_; }

    size_type l1jetSize() const { return l1jetIds_.size(); }
    const Vids& l1jetIds() const { return l1jetIds_; }
    const VRl1jet& l1jetRefs() const { return l1jetRefs_; }

    size_type l1etmissSize() const { return l1etmissIds_.size(); }
    const Vids& l1etmissIds() const { return l1etmissIds_; }
    const VRl1etmiss& l1etmissRefs() const { return l1etmissRefs_; }

    size_type l1hfringsSize() const { return l1hfringsIds_.size(); }
    const Vids& l1hfringsIds() const { return l1hfringsIds_; }
    const VRl1hfrings& l1hfringsRefs() const { return l1hfringsRefs_; }

    size_type pfjetSize() const { return pfjetIds_.size(); }
    const Vids& pfjetIds() const { return pfjetIds_; }
    const VRpfjet& pfjetRefs() const { return pfjetRefs_; }

    size_type pftauSize() const { return pftauIds_.size(); }
    const Vids& pftauIds() const { return pftauIds_; }
    const VRpftau& pftauRefs() const { return pftauRefs_; }

    size_type pfmetSize() const { return pfmetIds_.size(); }
    const Vids& pfmetIds() const { return pfmetIds_; }
    const VRpfmet& pfmetRefs() const { return pfmetRefs_; }

    size_type l1tmuonSize() const { return l1tmuonIds_.size(); }
    const Vids& l1tmuonIds() const { return l1tmuonIds_; }
    const VRl1tmuon& l1tmuonRefs() const { return l1tmuonRefs_; }

    size_type l1tmuonShowerSize() const { return l1tmuonShowerIds_.size(); }
    const Vids& l1tmuonShowerIds() const { return l1tmuonShowerIds_; }
    const VRl1tmuonShower& l1tmuonShowerRefs() const { return l1tmuonShowerRefs_; }

    size_type l1tegammaSize() const { return l1tegammaIds_.size(); }
    const Vids& l1tegammaIds() const { return l1tegammaIds_; }
    const VRl1tegamma& l1tegammaRefs() const { return l1tegammaRefs_; }

    size_type l1tjetSize() const { return l1tjetIds_.size(); }
    const Vids& l1tjetIds() const { return l1tjetIds_; }
    const VRl1tjet& l1tjetRefs() const { return l1tjetRefs_; }

    /* Phase-2 */

    size_type l1ttkmuonSize() const { return l1ttkmuonIds_.size(); }
    const Vids& l1ttkmuonIds() const { return l1ttkmuonIds_; }
    const VRl1ttkmuon& l1ttkmuonRefs() const { return l1ttkmuonRefs_; }

    size_type l1ttkeleSize() const { return l1ttkeleIds_.size(); }
    const Vids& l1ttkeleIds() const { return l1ttkeleIds_; }
    const VRl1ttkele& l1ttkeleRefs() const { return l1ttkeleRefs_; }

    size_type l1ttkemSize() const { return l1ttkemIds_.size(); }
    const Vids& l1ttkemIds() const { return l1ttkemIds_; }
    const VRl1ttkem& l1ttkemRefs() const { return l1ttkemRefs_; }

    size_type l1tpfjetSize() const { return l1tpfjetIds_.size(); }
    const Vids& l1tpfjetIds() const { return l1tpfjetIds_; }
    const VRl1tpfjet& l1tpfjetRefs() const { return l1tpfjetRefs_; }

    size_type l1tpftauSize() const { return l1tpftauIds_.size(); }
    const Vids& l1tpftauIds() const { return l1tpftauIds_; }
    const VRl1tpftau& l1tpftauRefs() const { return l1tpftauRefs_; }

    size_type l1thpspftauSize() const { return l1thpspftauIds_.size(); }
    const Vids& l1thpspftauIds() const { return l1thpspftauIds_; }
    const VRl1thpspftau& l1thpspftauRefs() const { return l1thpspftauRefs_; }

    size_type l1tpftrackSize() const { return l1tpftrackIds_.size(); }
    const Vids& l1tpftrackIds() const { return l1tpftrackIds_; }
    const VRl1tpftrack& l1tpftrackRefs() const { return l1tpftrackRefs_; }

    size_type l1tp2etsumSize() const { return l1tp2etsumIds_.size(); }
    const Vids& l1tp2etsumIds() const { return l1tp2etsumIds_; }
    const VRl1tp2etsum& l1tp2etsumRefs() const { return l1tp2etsumRefs_; }

    size_type l1ttauSize() const { return l1ttauIds_.size(); }
    const Vids& l1ttauIds() const { return l1ttauIds_; }
    const VRl1ttau& l1ttauRefs() const { return l1ttauRefs_; }

    size_type l1tetsumSize() const { return l1tetsumIds_.size(); }
    const Vids& l1tetsumIds() const { return l1tetsumIds_; }
    const VRl1tetsum& l1tetsumRefs() const { return l1tetsumRefs_; }
  };

  // picked up via argument dependent lookup, e-g- by boost::swap()
  inline void swap(TriggerRefsCollections& first, TriggerRefsCollections& second) { first.swap(second); }

}  // namespace trigger

#endif
