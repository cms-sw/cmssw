#ifndef PhysicsTools_PatUtils_interface_MuonVPlusJetsIDSelectionFunctor_h
#define PhysicsTools_PatUtils_interface_MuonVPlusJetsIDSelectionFunctor_h

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/Utilities/interface/Selector.h"

#include <iostream>

class MuonVPlusJetsIDSelectionFunctor : public Selector<pat::Muon> {

 public: // interface

  enum Version_t { SUMMER08, N_VERSIONS };

  MuonVPlusJetsIDSelectionFunctor( Version_t version,
				   double chi2 = 10.0,
				   double d0 = 0.2,
				   int nhits = 11,
				   double ecaliso = 4.0,
				   double hcaliso = 6.0,
				   double reliso = 0.05 ) :
  version_(version)
  {
    push_back("Chi2",      chi2   );
    push_back("D0",        d0     );
    push_back("NHits",     nhits  );
    push_back("ECalVeto",   ecaliso);
    push_back("HCalVeto",   hcaliso);
    push_back("RelIso",    reliso );

    set("Chi2");
    set("D0");
    set("NHits");
    set("ECalVeto");
    set("HCalVeto");
    set("RelIso");

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
    int nhits = static_cast<int>( muon.numberOfValidHits() );
    
    double ecalVeto = muon.isolationR03().emVetoEt;
    double hcalVeto = muon.isolationR03().hadVetoEt;
	
    double hcalIso = muon.hcalIso();
    double ecalIso = muon.ecalIso();
    double trkIso  = muon.trackIso();
    double pt      = muon.pt() ;

    double relIso = (ecalIso + hcalIso + trkIso) / pt;

    if ( norm_chi2     <  cut("Chi2",   double()) || ignoreCut("Chi2")    ) passCut(ret, "Chi2"   );
    if ( fabs(corr_d0) <  cut("D0",     double()) || ignoreCut("D0")      ) passCut(ret, "D0"     );
    if ( nhits         >= cut("NHits",  int()   ) || ignoreCut("NHits")   ) passCut(ret, "NHits"  );
    if ( hcalVeto      <  cut("HCalVeto",double())|| ignoreCut("HCalVeto")) passCut(ret, "HCalVeto");
    if ( ecalVeto      <  cut("ECalVeto",double())|| ignoreCut("ECalVeto")) passCut(ret, "ECalVeto");
    if ( relIso        <  cut("RelIso", double()) || ignoreCut("RelIso")  ) passCut(ret, "RelIso" );

    return (bool)ret;
  }
  
 private: // member variables
  
  Version_t version_;
  
};

#endif
