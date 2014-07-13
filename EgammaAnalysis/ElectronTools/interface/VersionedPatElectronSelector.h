#ifndef EgammaAnalysis_ElectronTools_interface_VersionedPatElectronSelector_h
#define EgammaAnalysis_ElectronTools_interface_VersionedPatElectronSelector_h

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/SelectorUtils/interface/VersionedSelector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>

class VersionedPatElectronSelector : public VersionedSelector<pat::Electron> {
 public:
 VersionedPatElectronSelector() : 
  VersionedSelector<pat::Electron>(),
    initialized_(false) {}
  
 VersionedPatElectronSelector( const edm::ParameterSet& parameters );
  
 void initialize( const edm::ParameterSet& parameters );

  // Allow for multiple definitions of the cuts.
 bool operator()(const pat::Electron&,pat::strbitset&); 
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
 bool operator()(const pat::Electron&,
		 edm::EventBase const&,
		 pat::strbitset&) override final;
#else
 bool operator()(const pat::Electron&,
		 edm::EventBase const&,
		 pat::strbitset&);
#endif
 using VersionedSelector<pat::Electron>::operator();
 
 private:
 bool initialized_; 
};

#endif
