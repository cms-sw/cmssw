#ifndef PhysicsTools_PatUtils_interface_MuonVPlusJetsIDSelectionFunctor_h
#define PhysicsTools_PatUtils_interface_MuonVPlusJetsIDSelectionFunctor_h

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <functional>

class MuonVPlusJetsIDSelectionFunctor : public std::unary_function<pat::Muon, bool>  {

 public: // interface

  enum Version_t { SUMMER08, N_VERSIONS };

 MuonVPlusJetsIDSelectionFunctor( Version_t version ) :
  version_(version)
  {
  }

  // Allow for multiple definitions of the cuts. 
  bool operator()( const pat::Muon & muon ) const 
  { 

    if ( version_ == SUMMER08 ) return summer08Cuts( muon );
    else {
      return false;
    }
  }

  // cuts based on craft 08 analysis. 
  bool summer08Cuts( const pat::Muon & muon) const
  {
    double norm_chi2 = muon.normChi2();
    double corr_d0 = muon.dB();
    unsigned int nhits = muon.numberOfValidHits();
	
    double hcalIso = muon.hcalIso();
    double ecalIso = muon.ecalIso();
    double trkIso  = muon.trackIso();
    double pt      = muon.pt() ;

    double relIso = (ecalIso + hcalIso + trkIso) / pt;

    // if the muon passes the event selection, add it to the output list
    if ( norm_chi2      < 10. &&
	 fabs(corr_d0)  < 0.2 &&
	 nhits          >= 11 &&
	 hcalIso        < 6.0 &&
	 ecalIso        < 4.0 &&
	 relIso         < 0.1 ) {
      return true;
    } else {
      return false;
    }
  }
  
 private: // member variables
  
  Version_t version_;
  
};

#endif
