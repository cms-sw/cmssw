#ifndef PhysicsTools_PatUtils_interface_ElectronVPlusJetsIDSelectionFunctor_h
#define PhysicsTools_PatUtils_interface_ElectronVPlusJetsIDSelectionFunctor_h

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/Utilities/interface/Selector.h"

class ElectronVPlusJetsIDSelectionFunctor : public Selector<pat::Electron>  {

 public: // interface

  enum Version_t { SUMMER08, N_VERSIONS };

 ElectronVPlusJetsIDSelectionFunctor( Version_t version ) :
  version_(version)
  {
    push_back("D0");
    push_back("RelIso");
  }

  // Allow for multiple definitions of the cuts. 
  bool operator()( const pat::Electron & electron ) const 
  { 

    if ( version_ == SUMMER08 ) return summer08Cuts( electron );
    else {
      return false;
    }
  }

  // cuts based on craft 08 analysis. 
  bool summer08Cuts( const pat::Electron & electron) const
  {

    double corr_d0 = electron.dB();
	
    double hcalIso = electron.hcalIso();
    double ecalIso = electron.ecalIso();
    double trkIso  = electron.trackIso();
    double pt      = electron.pt() ;
    
    double relIso = (ecalIso + hcalIso + trkIso) / pt;


    if ( fabs(corr_d0) > 0.2 && (*this)["D0"]      ) return false;
    if ( relIso  > 0.1       && (*this)["RelIso"]  ) return false;

    return true;
  }
  
 private: // member variables
  
  Version_t version_;
  
};

#endif
