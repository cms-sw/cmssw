#ifndef EgammaAnalysis_ElectronTools_interface_VersionedGsfElectronSelector_h
#define EgammaAnalysis_ElectronTools_interface_VersionedGsfElectronSelector_h

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/SelectorUtils/interface/VersionedSelector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>

class VersionedGsfElectronSelector : public VersionedSelector<reco::GsfElectronRef> {
 public:
 VersionedGsfElectronSelector() : 
  VersionedSelector<reco::GsfElectronRef>(),
    initialized_(false) {}
  
 VersionedGsfElectronSelector( const edm::ParameterSet& parameters );
  
 void initialize( const edm::ParameterSet& parameters );

  // Allow for multiple definitions of the cuts.
 bool operator()(const reco::GsfElectronRef&,pat::strbitset&); 
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
 bool operator()(const reco::GsfElectronRef&,
		 edm::EventBase const&,
		 pat::strbitset&) override final;
#else
 bool operator()(const reco::GsfElectronRef&,
		 edm::EventBase const&,
		 pat::strbitset&);
#endif

 using VersionedSelector<reco::GsfElectronRef>::operator();
 
 private:
 bool initialized_; 
};

#endif
