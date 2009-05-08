#ifndef TopObjects_StGenEvent_h
#define TopObjects_StGenEvent_h

#include "AnalysisDataFormats/TopObjects/interface/TopGenEvent.h"


// ----------------------------------------------------------------------
// derived class for: 
//
//  * StGenEvent
//
//  the structure holds reference information to the generator particles 
//  of the decay chains for each top quark and of the initial partons 
//  and provides access and administration;  the derived class contains 
//  a few additional getters with respect to its base class
// ----------------------------------------------------------------------

class StGenEvent: public TopGenEvent {

 public:
  /// empty constructor  
  StGenEvent();
  /// default constructor
  StGenEvent(reco::GenParticleRefProd&, reco::GenParticleRefProd&);
  /// default destructor
  virtual ~StGenEvent();

  /// return single lepton if available; 0 else
  const reco::GenParticle* singleLepton() const;
  /// return single neutrino if available; 0 else
  const reco::GenParticle* singleNeutrino() const;
  /// return single W
  const reco::GenParticle* singleW() const;
  /// return single Top
  const reco::GenParticle* singleTop() const;
  /// return decay b
  const reco::GenParticle* decayB() const;
  /// return associated b 
  const reco::GenParticle* associatedB() const;
};

#endif
