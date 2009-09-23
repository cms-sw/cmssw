#ifndef PhysicsTools_PatUtils_interface_ElectronVPlusJetsIDSelectionFunctor_h
#define PhysicsTools_PatUtils_interface_ElectronVPlusJetsIDSelectionFunctor_h

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <functional>

class ElectronVPlusJetsIDSelectionFunctor : public std::unary_function<pat::Electron, bool>  {

 public: // interface

  enum Version_t { SUMMER08, N_VERSIONS };

 ElectronVPlusJetsIDSelectionFunctor( Version_t version ) :
  version_(version)
  {
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

    // if the electron passes the event selection, add it to the output list
    if ( fabs(corr_d0)  < 0.2 &&
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
