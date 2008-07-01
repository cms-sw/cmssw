#ifndef HepMCCandidate_FlavorHistory_h
#define HepMCCandidate_FlavorHistory_h

/** \class reco::FlavorHistory
 *
 * Stores information about the flavor history of a parton
 *
 * \author: Stephen Mrenna (FNAL), Salvatore Rappoccio (JHU)
 *
 */



// Identify the ancestry of the b Quark
// Mother               Origin
// ======               =======
// incoming quarks      ISR, likely gluon splitting
//   light flavor
// incoming quarks      ISR, likely flavor excitation
//   heavy flavor           
// outgoing quark       FSR
//   light flavor
// outgoing quark       Matrix Element b       
//   heavy flavor
//     no mother
// outgoing quark       Resonance b (e.g. top quark decay)
//   heavy flavor
//     mother
// outgoing resonance   Resonance b (e.g. Higgs decay)


#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

namespace reco {


class FlavorHistory {
public:

  enum FLAVOR_T { FLAVOR_NULL=0,
		  FLAVOR_GS,
		  FLAVOR_EXC,
		  FLAVOR_ME,
		  FLAVOR_DECAY,
		  N_FLAVOR_TYPES };

  static const int  gluonId=21;
  static const int  tQuarkId=6;
  static const int  bQuarkId=5;
  static const int  cQuarkId=4;
  

  FlavorHistory(); 
  FlavorHistory( FLAVOR_T flavorSource,
		 reco::CandidatePtr const & parton,
		 reco::CandidatePtr const & progenitor,
		 reco::CandidatePtr const & sister );
  FlavorHistory( FLAVOR_T flavorSource,
		 edm::Handle<edm::View<reco::Candidate> > h_partons,
		 int iparton,
		 int iprogenitor,
		 int isister);
  FlavorHistory( FLAVOR_T flavorSource,
		 edm::Handle<reco::CandidateCollection > h_partons,
		 int iparton,
		 int iprogenitor,
		 int isister);
  ~FlavorHistory(){}


  // Accessors
  FLAVOR_T       flavorSource          () const { return flavorSource_; }
  bool           hasParton             () const { return parton_.key() > 0; }
  bool           hasSister             () const { return sister_.key() > 0; }
  bool           hasProgenitor         () const { return progenitor_.key() > 0 ;}
  const reco::CandidatePtr & parton    () const { return parton_; }
  const reco::CandidatePtr & sister    () const { return sister_; }
  const reco::CandidatePtr & progenitor() const { return progenitor_; }

  // Operators for sorting and keys
  bool operator< ( FlavorHistory const & right ) {
    return parton_.key() < right.parton_.key();
  }
  bool operator> ( FlavorHistory const & right ) {
    return parton_.key() > right.parton_.key();
  }
  bool operator== ( FlavorHistory const & right ) {
    return parton_.key() == right.parton_.key();
  }
  
  

protected:
  FLAVOR_T              flavorSource_;
  reco::CandidatePtr    parton_;
  reco::CandidatePtr    progenitor_;
  reco::CandidatePtr    sister_;
};

}

#endif
