// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: L1OffsetCorrector.h,v 1.1 2007/12/08 01:55:42 fedor Exp $
//
// Level3 Absolute Corrector
//
#ifndef L1OffsetCorrector_h
#define L1OffsetCorrector_h

#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleL1OffsetCorrector.h"


/// classes declaration
namespace edm {
  class ParameterSet;
}

class SimpleL1OffsetCorrector;

class L1OffsetCorrector : public JetCorrector {
 public:
  L1OffsetCorrector (const edm::ParameterSet& fParameters);
  virtual ~L1OffsetCorrector ();

  /// apply correction using Jet information only
  virtual double correction (const LorentzVector& fJet) const;
  /// apply correction using Jet information only
  virtual double correction (const reco::Jet& fJet) const;

  /// if correction needs event information
   virtual bool eventRequired () const {return false;} 

 private:
  SimpleL1OffsetCorrector* mSimpleCorrector;
};

#endif
