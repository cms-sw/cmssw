#ifndef EgammaAnalysis_ElectronTools_interface_VersionedGsfElectronSelector_h
#define EgammaAnalysis_ElectronTools_interface_VersionedGsfElectronSelector_h

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/SelectorUtils/interface/VersionedSelector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>

class VersionedGsfElectronSelector : public VersionedSelector<reco::GsfElectron> {
  
 public: // interface
  
 bool verbose_;  
 enum Version_t { SPRING11, N_VERSIONS };

 VersionedGsfElectronSelector() : VersionedSelector<reco::GsfElectron>( ) {}
  
 VersionedGsfElectronSelector( edm::ParameterSet const & parameters );
  
  void initialize( Version_t version,
		   double mva = 0.4,
		   double d0 = 0.02,
		   int nMissingHits = 1,
		   std::string eidUsed = "eidTightMC",
		   bool convRej = true,
		   double pfiso = 0.15 );

  // Allow for multiple definitions of the cuts.
  bool operator()( const reco::GsfElectron & electron, pat::strbitset & ret ) {    
    return ( version_ == SPRING11 && spring11Cuts(electron, ret) );
  }

  using VersionedSelector<reco::GsfElectron>::operator();

  // cuts based on top group L+J synchronization exercise
  bool spring11Cuts( const reco::GsfElectron & electron, pat::strbitset & ret);

 private: // member variables

  Version_t version_;

  index_type indexID;
  index_type indexMaxMissingHits_;
  index_type indexD0_;
  index_type indexConvRej_;
  index_type indexPFIso_;
  index_type indexMVA_;
  index_type indexElectronId_;

  std::string electronIDvalue_;
};

#endif
