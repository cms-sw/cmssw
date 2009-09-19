#ifndef PhysicsTools_PatUtils_interface_MuonVPlusJetsIDSelectionFunctor_h
#define PhysicsTools_PatUtils_interface_MuonVPlusJetsIDSelectionFunctor_h

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/Utilities/interface/Selector.h"

class MuonVPlusJetsIDSelectionFunctor : public Selector<pat::Muon> {

 public: // interface

  enum Version_t { SUMMER08, N_VERSIONS };

 MuonVPlusJetsIDSelectionFunctor( Version_t version ) :
  version_(version)
  {
    push_back("Chi2");
    push_back("D0");
    push_back("NHits");
    push_back("ECalIso");
    push_back("HCalIso");
    push_back("RelIso");

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

    if ( norm_chi2 >= 10.0   && (*this)["Chi2"]    ) return false;
    if ( fabs(corr_d0) > 0.2 && (*this)["D0"]      ) return false;
    if ( nhits < 11          && (*this)["NHits"]   ) return false;
    if ( hcalIso > 6.0       && (*this)["HCalIso"] ) return false;
    if ( ecalIso > 6.0       && (*this)["ECalIso"] ) return false;
    if ( relIso  > 0.1       && (*this)["RelIso"]  ) return false;

    return true;
  }
  
 private: // member variables
  
  Version_t version_;
  
};

#endif
