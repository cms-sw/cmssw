// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: L2RelativeCorrector.h,v 1.1.2.1 2007/11/20 21:13:30 fedor Exp $
//
// Level2 Relative Corrector
//
#ifndef L2RelativeCorrector_h
#define L2RelativeCorrector_h

#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleL2RelativeCorrector.h"


/// classes declaration
namespace edm {
  class ParameterSet;
}

class SimpleL2RelativeCorrector;

class L2RelativeCorrector : public JetCorrector {
 public:
  L2RelativeCorrector (const edm::ParameterSet& fParameters);
  virtual ~L2RelativeCorrector ();

  /// apply correction using Jet information only
  virtual double correction (const LorentzVector& fJet) const;
  /// apply correction using Jet information only
  virtual double correction (const reco::Jet& fJet) const;

  /// if correction needs event information
   virtual bool eventRequired () const {return false;} 

 private:
  SimpleL2RelativeCorrector* mSimpleCorrector;
};

#endif
