//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: L4EMFCorrector.h,v 1.3 2007/03/30 23:47:55 fedor Exp $
//
// Level4 EMF Corrector
//
#ifndef L4EMFCorrector_h
#define L4EMFCorrector_h

#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleL4EMFCorrector.h"


/// classes declaration
namespace edm {
  class ParameterSet;
}

class SimpleL4EMFCorrector;

class L4EMFCorrector : public JetCorrector {
 public:
  L4EMFCorrector (const edm::ParameterSet& fParameters);
  virtual ~L4EMFCorrector ();

  /// apply correction using Jet information only
  virtual double correction (const LorentzVector& fJet) const;

  /// apply correction using all event information !!!->this is huck to retrofit 1.6.X
   virtual double correction (const reco::Jet& fJet, const edm::Event& fEvent, const edm::EventSetup& fSetup) const;

  /// if correction needs event information
   virtual bool eventRequired () const {return false;} 

 private:
  SimpleL4EMFCorrector* mSimpleCorrector;
};

#endif
