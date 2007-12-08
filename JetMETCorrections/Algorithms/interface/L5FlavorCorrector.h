//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: L5FlavorCorrector.h,v 1.1.2.1 2007/11/17 00:53:23 fedor Exp $
//
//  L5 Jet flavor Corrector
//
#ifndef L5FlavorCorrector_h
#define L5FlavorCorrector_h

#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleL5FlavorCorrector.h"


/// classes declaration
namespace edm {
  class ParameterSet;
}

class SimpleL5FlavorCorrector;

class L5FlavorCorrector : public JetCorrector {
 public:
  L5FlavorCorrector (const edm::ParameterSet& fParameters);
  virtual ~L5FlavorCorrector ();

  /// apply correction using Jet information only
  virtual double correction (const LorentzVector& fJet) const;
  /// apply correction using Jet information only
  virtual double correction (const reco::Jet& fJet) const;

  /// if correction needs event information
   virtual bool eventRequired () const {return false;} 

 private:
  SimpleL5FlavorCorrector* mSimpleCorrector;
};

#endif
