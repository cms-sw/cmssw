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
#include <iostream>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHTrackReference {

 public:

  typedef pat::PackedCandidate candidate;

  /** Constructor
   */
  BPHTrackReference();

  /** Destructor
   */
  ~BPHTrackReference();

  /** Operations
   */
  /// get associated reco::Track calling a sequence of functions
  /// until a track is found; the list of functions to call is given
  /// as a string where each character identify a function:
  /// c :  reco ::       Candidate :: get<reco::TrackRef> ()
  /// f :  reco ::     PFCandidate ::           trackRef  ()
  /// h :   pat :: GenericParticle ::           track     ()
  /// b :  reco ::       Candidate ::       bestTrack     ()
  /// p :   pat :: PackedCandidate ::     pseudoTrack     ()
  /// m :   pat ::            Muon ::     pfCandidateRef  ()
  /// n :   pat ::            Muon ::   muonBestTrack     ()
  /// i :   pat ::            Muon ::      innerTrack     ()
  /// g :   pat ::            Muon ::     globalTrack     ()
  /// s :   pat ::            Muon ::      standAloneMuon ()
  /// e :   pat ::        Electron ::     pfCandidateRef  ()
  static const reco::Track* getTrack( const reco::Candidate& rc,
                            const char* modeList = "cfhbpmnigse",
                                  char* modeFlag = 0 ) {
    if ( rc.charge() == 0 ) return 0;
    const char* mptr = modeList;
    char c;
    if ( modeFlag == 0 ) modeFlag = &c;
    char& mode = *modeFlag;
    const reco::Track* tkp = 0;
    while ( ( mode = *mptr++ ) ) {
      switch ( mode ) {
      case 'c': if ( ( tkp = getFromRC( rc ) ) != 0 ) return tkp; break;
      case 'f': if ( ( tkp = getFromPF( rc ) ) != 0 ) return tkp; break;
      case 'h': if ( ( tkp = getFromGP( rc ) ) != 0 ) return tkp; break;
      case 'b': if ( ( tkp = getFromBT( rc ) ) != 0 ) return tkp; break;
      case 'p': if ( ( tkp = getFromPC( rc ) ) != 0 ) return tkp; break;
      case 'm': if ( ( tkp = getMuonPF( rc ) ) != 0 ) return tkp; break;
      case 'n': if ( ( tkp = getMuonBT( rc ) ) != 0 ) return tkp; break;
      case 'i': if ( ( tkp = getMuonIT( rc ) ) != 0 ) return tkp; break;
      case 'g': if ( ( tkp = getMuonGT( rc ) ) != 0 ) return tkp; break;
      case 's': if ( ( tkp = getMuonSA( rc ) ) != 0 ) return tkp; break;
      case 'e': if ( ( tkp = getElecPF( rc ) ) != 0 ) return tkp; break;
      }
    }
    return 0;
  }

  static const reco::Track* getFromRC( const reco::Candidate& rc ) {
//    std::cout << "getFromRC" << std::endl;
    try {
      const reco::TrackRef& tkr = rc.get<reco::TrackRef>();
      if ( tkr.isNonnull() && tkr.isAvailable() ) return tkr.get();
    }
    catch ( edm::Exception e ) {
    }
    return 0;
  }
  static const reco::Track* getFromPF( const reco::Candidate& rc ) {
//    std::cout << "getFromPF" << std::endl;
    const reco::PFCandidate* pf =
          dynamic_cast<const reco::PFCandidate*>( &rc );
    if ( pf == 0 ) return 0;
    try {
      const reco::TrackRef& tkr = pf->trackRef();
      if ( tkr.isNonnull() && tkr.isAvailable() ) return tkr.get();
    }
    catch ( edm::Exception e ) {
    }
    return 0;
  }
  static const reco::Track* getFromGP( const reco::Candidate& rc ) {
//    std::cout << "getFromGC" << std::endl;
    const pat::GenericParticle* gp =
        dynamic_cast<const pat::GenericParticle*>( &rc );
    if ( gp == 0 ) return 0;
    try {
      const reco::TrackRef& tkr = gp->track();
      if ( tkr.isNonnull() && tkr.isAvailable() ) return tkr.get();
    }
    catch ( edm::Exception e ) {
    }
    return 0;
  }
  static const reco::Track* getFromBT( const reco::Candidate& rc ) {
//    std::cout << "getFromBT" << std::endl;
    try {
      const reco::Track* trk = rc.bestTrack();
      return trk;
    }
    catch ( edm::Exception e ) {
    }
    return 0;
  }
  static const reco::Track* getFromPC( const reco::Candidate& rc ) {
//    std::cout << "getFromPC" << std::endl;
    const pat::PackedCandidate* pp =
        dynamic_cast<const pat::PackedCandidate*>( &rc );
    if ( pp == 0 ) return 0;
    try {
      const reco::Track* trk = &pp->pseudoTrack();
      return trk;
    }
    catch ( edm::Exception e ) {
    }
    return 0;
  }
  static const reco::Track* getMuonPF( const reco::Candidate& rc ) {
//    std::cout << "getMuonPF" << std::endl;
    const pat::Muon* mu = dynamic_cast<const pat::Muon*>( &rc );
    if ( mu == 0 ) return 0;
    return getMuonPF( mu );
  }
  static const reco::Track* getMuonPF( const pat::Muon* mu ) {
    try {
      const reco::PFCandidateRef& pcr = mu->pfCandidateRef();
      if ( pcr.isNonnull() && pcr.isAvailable() ) {
        const reco::TrackRef& tkr = pcr->trackRef();
        if ( tkr.isNonnull() && tkr.isAvailable() ) return tkr.get();
      }
    }
    catch ( edm::Exception e ) {
    }
    return 0;
  }
  static const reco::Track* getMuonBT( const reco::Candidate& rc ) {
//    std::cout << "getMuonBT" << std::endl;
    const reco::Muon* mu = dynamic_cast<const reco::Muon*>( &rc );
    if ( mu == 0 ) return 0;
    return getMuonBT( mu );
  }
  static const reco::Track* getMuonBT( const reco::Muon* mu ) {
    try {
      const reco::TrackRef& tkr = mu->muonBestTrack();
      if ( tkr.isNonnull() && tkr.isAvailable() ) return tkr.get();
    }
    catch ( edm::Exception e ) {
    }
    return 0;
  }
  static const reco::Track* getMuonIT( const reco::Candidate& rc ) {
//    std::cout << "getMuonIT" << std::endl;
    const pat::Muon* mu = dynamic_cast<const pat::Muon*>( &rc );
    if ( mu == 0 ) return 0;
    return getMuonIT( mu );
  }
  static const reco::Track* getMuonIT( const pat::Muon* mu ) {
    if ( !mu->isTrackerMuon() ) return 0;
    try {
      const reco::TrackRef& mit = mu->innerTrack();
      if ( mit.isNonnull() && mit.isAvailable() ) return mit.get();
    }
    catch ( edm::Exception e ) {
    }
    return 0;
  }
  static const reco::Track* getMuonGT( const reco::Candidate& rc ) {
//    std::cout << "getMuonGT" << std::endl;
    const pat::Muon* mu = dynamic_cast<const pat::Muon*>( &rc );
    if ( mu == 0 ) return 0;
    return getMuonGT( mu );
  }
  static const reco::Track* getMuonGT( const pat::Muon* mu ) {
    if ( !mu->isGlobalMuon() ) return 0;
    try {
      const reco::TrackRef& mgt = mu->globalTrack();
      if ( mgt.isNonnull() && mgt.isAvailable() ) return mgt.get();
    }
    catch ( edm::Exception e ) {
    }
    return 0;
  }
  static const reco::Track* getMuonSA( const reco::Candidate& rc ) {
//    std::cout << "getMuonGT" << std::endl;
    const pat::Muon* mu = dynamic_cast<const pat::Muon*>( &rc );
    if ( mu == 0 ) return 0;
    return getMuonSA( mu );
  }
  static const reco::Track* getMuonSA( const pat::Muon* mu ) {
    if ( !mu->isStandAloneMuon() ) return 0;
    try {
      const reco::TrackRef& msa = mu->standAloneMuon();
      if ( msa.isNonnull() && msa.isAvailable() ) return msa.get();
    }
    catch ( edm::Exception e ) {
    }
    return 0;
  }
  static const reco::Track* getElecPF( const reco::Candidate& rc ) {
//    std::cout << "getElecPF" << std::endl;
    const pat::Electron* el = dynamic_cast<const pat::Electron*>( &rc );
    if ( el == 0 ) return 0;
    return getElecPF( el );
  }
  static const reco::Track* getElecPF( const pat::Electron* el ) {
    try {
      const reco::PFCandidateRef& pcr = el->pfCandidateRef();
      if ( pcr.isNonnull() && pcr.isAvailable() ) {
        const reco::TrackRef& tkr = pcr->trackRef();
        if ( tkr.isNonnull() && tkr.isAvailable() ) return tkr.get();
      }
    }
    catch ( edm::Exception e ) {
    }
    return 0;
  }

 private:

  // private copy and assigment constructors
  BPHTrackReference           ( const BPHTrackReference& x );
  BPHTrackReference& operator=( const BPHTrackReference& x );

};


#endif

