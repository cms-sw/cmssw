// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: L3AbsoluteCorrector.h,v 1.1.2.1 2007/11/20 21:13:30 fedor Exp $
//
// Level3 Absolute Corrector
//
#ifndef L3AbsoluteCorrector_h
#define L3AbsoluteCorrector_h

#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleL3AbsoluteCorrector.h"


/// classes declaration
namespace edm {
  class ParameterSet;
}

class SimpleL3AbsoluteCorrector;

class L3AbsoluteCorrector : public JetCorrector {
 public:
  L3AbsoluteCorrector (const edm::ParameterSet& fParameters);
  virtual ~L3AbsoluteCorrector ();

  /// apply correction using Jet information only
  virtual double correction (const LorentzVector& fJet) const;
  /// apply correction using Jet information only
  virtual double correction (const reco::Jet& fJet) const;

  /// if correction needs event information
   virtual bool eventRequired () const {return false;} 

 private:
  SimpleL3AbsoluteCorrector* mSimpleCorrector;
};

#endif
