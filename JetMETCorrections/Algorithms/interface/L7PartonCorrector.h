//
// Original Author:  Attilio Santocchia Feb 28, 2008
//
//  L7 Jet Parton Corrector
//
#ifndef L7PartonCorrector_h
#define L7PartonCorrector_h

#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleL7PartonCorrector.h"


/// classes declaration
namespace edm {
  class ParameterSet;
}

class SimpleL7PartonCorrector;

class L7PartonCorrector : public JetCorrector {
 public:
  L7PartonCorrector (const edm::ParameterSet& fParameters);
  virtual ~L7PartonCorrector ();

  /// apply correction using Jet information only
  virtual double correction (const LorentzVector& fJet) const;
  /// apply correction using Jet information only
  virtual double correction (const reco::Jet& fJet) const;

  /// if correction needs event information
   virtual bool eventRequired () const {return false;} 

 private:
  SimpleL7PartonCorrector* mSimpleCorrector;
};

#endif
