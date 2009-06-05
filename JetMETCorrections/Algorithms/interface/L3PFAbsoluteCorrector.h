// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: L3AbsoluteCorrector.h,v 1.1 2007/12/08 01:55:42 fedor Exp $
//
// Level3 Absolute Corrector
//
#ifndef L3PFAbsoluteCorrector_h
#define L3PFAbsoluteCorrector_h

#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleL3PFAbsoluteCorrector.h"


/// classes declaration
namespace edm {
  class ParameterSet;
}

class SimpleL3PFAbsoluteCorrector;

class L3PFAbsoluteCorrector : public JetCorrector {
 public:
  L3PFAbsoluteCorrector (const edm::ParameterSet& fParameters);
  virtual ~L3PFAbsoluteCorrector ();

  /// apply correction using Jet information only
  virtual double correction (const LorentzVector& fJet) const;
  /// apply correction using Jet information only
  virtual double correction (const reco::Jet& fJet) const;

  /// if correction needs event information
   virtual bool eventRequired () const {return false;} 

 private:
  SimpleL3PFAbsoluteCorrector* mSimpleCorrector;
};

#endif
