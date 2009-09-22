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
    push_back("Chi2",     10.0);
    push_back("D0",        0.2);
    push_back("NHits",      11);
    push_back("ECalIso",   6.0);
    push_back("HCalIso",   4.0);
    push_back("RelIso",    0.1);

  }

  // Allow for multiple definitions of the cuts. 
  bool operator()( const pat::Muon & muon, std::strbitset & ret ) 
  { 

    if ( version_ == SUMMER08 ) return summer08Cuts( muon, ret );
    else {
      return false;
    }
  }

  // cuts based on craft 08 analysis. 
  bool summer08Cuts( const pat::Muon & muon, std::strbitset & ret)
  {
    double norm_chi2 = muon.normChi2();
    double corr_d0 = muon.dB();
    unsigned int nhits = muon.numberOfValidHits();
	
    double hcalIso = muon.hcalIso();
    double ecalIso = muon.ecalIso();
    double trkIso  = muon.trackIso();
    double pt      = muon.pt() ;

    double relIso = (ecalIso + hcalIso + trkIso) / pt;

    if ( norm_chi2     <  cut("Chi2",   double()) || !(*this)["Chi2"]    ) passCut(ret, "Chi2"   );
    if ( fabs(corr_d0) <  cut("D0",     double()) || !(*this)["D0"]      ) passCut(ret, "D0"     );
    if ( nhits         >= cut("NHits",  int()   ) || !(*this)["NHits"]   ) passCut(ret, "NHits"  );
    if ( hcalIso       <  cut("HCalIso",double()) || !(*this)["HCalIso"] ) passCut(ret, "HCalIso");
    if ( ecalIso       <  cut("ECalIso",double()) || !(*this)["ECalIso"] ) passCut(ret, "ECalIso");
    if ( relIso        <  cut("RelIso", double()) || !(*this)["RelIso"]  ) passCut(ret, "RelIso" );

    return true;
  }
  
 private: // member variables
  
  Version_t version_;
  
};

#endif
