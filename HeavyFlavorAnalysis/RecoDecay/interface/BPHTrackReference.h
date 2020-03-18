#ifndef HeavyFlavorAnalysis_RecoDecay_BPHTrackReference_h
#define HeavyFlavorAnalysis_RecoDecay_BPHTrackReference_h
/** \class BPHTrackReference
 *
 *  Description: 
 *     class to have uniform access to reco::Track
 *     for different particle objects
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/PatCandidates/interface/GenericParticle.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"

//---------------
// C++ Headers --
//---------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHTrackReference {
public:
  typedef pat::PackedCandidate candidate;

  /** Only static functions, no data member
   */

  /** Operations
   */
  /// get associated reco::Track calling a sequence of functions
  /// until a track is found; the list of functions to call is given
  /// as a string where each character identify a function:
  /// c :  reco ::       Candidate ::        get<reco::TrackRef> ()
  /// f :  reco ::     PFCandidate ::                  trackRef  ()
  /// h :   pat :: GenericParticle ::                  track     ()
  /// b :  reco ::       Candidate ::              bestTrack     ()
  /// p :   pat :: PackedCandidate ::            pseudoTrack     ()
  /// m :   pat ::            Muon ::pfCandidateRef()::trackRef  ()
  /// n :   pat ::            Muon ::          muonBestTrack     ()
  /// i :   pat ::            Muon ::             innerTrack     ()
  /// g :   pat ::            Muon ::            globalTrack     ()
  /// s :   pat ::            Muon ::             standAloneMuon ()
  /// e :   pat ::        Electron ::pfCandidateRef()::trackRef  ()
  /// t :   pat ::        Electron ::        closestCtfTrackRef  ()
  static const reco::Track* getTrack(const reco::Candidate& rc,
                                     const char* modeList = "cfhbpmnigset",
                                     char* modeFlag = nullptr) {
    if (rc.charge() == 0)
      return nullptr;
    const char* mptr = modeList;
    char c;
    if (modeFlag == nullptr)
      modeFlag = &c;
    char& mode = *modeFlag;
    const reco::Track* tkp = nullptr;
    while ((mode = *mptr++)) {
      switch (mode) {
        case 'c':
          if ((tkp = getFromRC(rc)) != nullptr)
            return tkp;
          break;
        case 'f':
          if ((tkp = getFromPF(rc)) != nullptr)
            return tkp;
          break;
        case 'h':
          if ((tkp = getFromGP(rc)) != nullptr)
            return tkp;
          break;
        case 'b':
          if ((tkp = getFromBT(rc)) != nullptr)
            return tkp;
          break;
        case 'p':
          if ((tkp = getFromPC(rc)) != nullptr)
            return tkp;
          break;
        case 'm':
          if ((tkp = getMuonPF(rc)) != nullptr)
            return tkp;
          break;
        case 'n':
          if ((tkp = getMuonBT(rc)) != nullptr)
            return tkp;
          break;
        case 'i':
          if ((tkp = getMuonIT(rc)) != nullptr)
            return tkp;
          break;
        case 'g':
          if ((tkp = getMuonGT(rc)) != nullptr)
            return tkp;
          break;
        case 's':
          if ((tkp = getMuonSA(rc)) != nullptr)
            return tkp;
          break;
        case 'e':
          if ((tkp = getElecPF(rc)) != nullptr)
            return tkp;
          break;
        case 't':
          if ((tkp = getElecTC(rc)) != nullptr)
            return tkp;
          break;
      }
    }
    return nullptr;
  }

  static const reco::Track* getFromRC(const reco::Candidate& rc) {
    try {
      const reco::TrackRef& tkr = rc.get<reco::TrackRef>();
      if (tkr.isNonnull() && tkr.isAvailable())
        return tkr.get();
    } catch (edm::Exception const&) {
    }
    return nullptr;
  }
  static const reco::Track* getFromPF(const reco::Candidate& rc) {
    const reco::PFCandidate* pf = dynamic_cast<const reco::PFCandidate*>(&rc);
    if (pf == nullptr)
      return nullptr;
    try {
      const reco::TrackRef& tkr = pf->trackRef();
      if (tkr.isNonnull() && tkr.isAvailable())
        return tkr.get();
    } catch (edm::Exception const&) {
    }
    return nullptr;
  }
  static const reco::Track* getFromGP(const reco::Candidate& rc) {
    const pat::GenericParticle* gp = dynamic_cast<const pat::GenericParticle*>(&rc);
    if (gp == nullptr)
      return nullptr;
    try {
      const reco::TrackRef& tkr = gp->track();
      if (tkr.isNonnull() && tkr.isAvailable())
        return tkr.get();
    } catch (edm::Exception const&) {
    }
    return nullptr;
  }
  static const reco::Track* getFromBT(const reco::Candidate& rc) {
    try {
      const reco::Track* trk = rc.bestTrack();
      return trk;
    } catch (edm::Exception const&) {
    }
    return nullptr;
  }
  static const reco::Track* getFromPC(const reco::Candidate& rc) {
    const pat::PackedCandidate* pp = dynamic_cast<const pat::PackedCandidate*>(&rc);
    if (pp == nullptr)
      return nullptr;
    try {
      const reco::Track* trk = &pp->pseudoTrack();
      return trk;
    } catch (edm::Exception const&) {
    }
    return nullptr;
  }
  static const reco::Track* getMuonPF(const reco::Candidate& rc) {
    const pat::Muon* mu = dynamic_cast<const pat::Muon*>(&rc);
    if (mu == nullptr)
      return nullptr;
    return getMuonPF(mu);
  }
  static const reco::Track* getMuonPF(const pat::Muon* mu) {
    try {
      const reco::PFCandidateRef& pcr = mu->pfCandidateRef();
      if (pcr.isNonnull() && pcr.isAvailable()) {
        const reco::TrackRef& tkr = pcr->trackRef();
        if (tkr.isNonnull() && tkr.isAvailable())
          return tkr.get();
      }
    } catch (edm::Exception const&) {
    }
    return nullptr;
  }
  static const reco::Track* getMuonBT(const reco::Candidate& rc) {
    const reco::Muon* mu = dynamic_cast<const reco::Muon*>(&rc);
    if (mu == nullptr)
      return nullptr;
    return getMuonBT(mu);
  }
  static const reco::Track* getMuonBT(const reco::Muon* mu) {
    try {
      const reco::TrackRef& tkr = mu->muonBestTrack();
      if (tkr.isNonnull() && tkr.isAvailable())
        return tkr.get();
    } catch (edm::Exception const&) {
    }
    return nullptr;
  }
  static const reco::Track* getMuonIT(const reco::Candidate& rc) {
    const pat::Muon* mu = dynamic_cast<const pat::Muon*>(&rc);
    if (mu == nullptr)
      return nullptr;
    return getMuonIT(mu);
  }
  static const reco::Track* getMuonIT(const pat::Muon* mu) {
    if (!mu->isTrackerMuon())
      return nullptr;
    try {
      const reco::TrackRef& mit = mu->innerTrack();
      if (mit.isNonnull() && mit.isAvailable())
        return mit.get();
    } catch (edm::Exception const&) {
    }
    return nullptr;
  }
  static const reco::Track* getMuonGT(const reco::Candidate& rc) {
    const pat::Muon* mu = dynamic_cast<const pat::Muon*>(&rc);
    if (mu == nullptr)
      return nullptr;
    return getMuonGT(mu);
  }
  static const reco::Track* getMuonGT(const pat::Muon* mu) {
    if (!mu->isGlobalMuon())
      return nullptr;
    try {
      const reco::TrackRef& mgt = mu->globalTrack();
      if (mgt.isNonnull() && mgt.isAvailable())
        return mgt.get();
    } catch (edm::Exception const&) {
    }
    return nullptr;
  }
  static const reco::Track* getMuonSA(const reco::Candidate& rc) {
    const pat::Muon* mu = dynamic_cast<const pat::Muon*>(&rc);
    if (mu == nullptr)
      return nullptr;
    return getMuonSA(mu);
  }
  static const reco::Track* getMuonSA(const pat::Muon* mu) {
    if (!mu->isStandAloneMuon())
      return nullptr;
    try {
      const reco::TrackRef& msa = mu->standAloneMuon();
      if (msa.isNonnull() && msa.isAvailable())
        return msa.get();
    } catch (edm::Exception const&) {
    }
    return nullptr;
  }
  static const reco::Track* getElecPF(const reco::Candidate& rc) {
    const pat::Electron* el = dynamic_cast<const pat::Electron*>(&rc);
    if (el == nullptr)
      return nullptr;
    return getElecPF(el);
  }
  static const reco::Track* getElecPF(const pat::Electron* el) {
    try {
      const reco::PFCandidateRef& pcr = el->pfCandidateRef();
      if (pcr.isNonnull() && pcr.isAvailable()) {
        const reco::TrackRef& tkr = pcr->trackRef();
        if (tkr.isNonnull() && tkr.isAvailable())
          return tkr.get();
      }
    } catch (edm::Exception const&) {
    }
    return nullptr;
  }
  static const reco::Track* getElecTC(const reco::Candidate& rc) {
    const pat::Electron* el = dynamic_cast<const pat::Electron*>(&rc);
    if (el == nullptr)
      return nullptr;
    return getElecTC(el);
  }
  static const reco::Track* getElecTC(const pat::Electron* el) {
    try {
      // Return the ctftrack closest to the electron
      const reco::TrackRef& tkr = el->closestCtfTrackRef();
      if (tkr.isNonnull() && tkr.isAvailable())
        return tkr.get();
    } catch (edm::Exception const&) {
    }
    return nullptr;
  }
};

#endif
