//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: ZSPJetCorrector.h,v 1.3 2007/12/08 01:55:42 fedor Exp $
//
// MC Jet Corrector
//
#ifndef ZSPJetCorrector_h
#define ZSPJetCorrector_h

#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleZSPJetCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleL1OffsetCorrector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

/// classes declaration
namespace edm {
  class ParameterSet;
}

class SimpleZSPJetCorrector;

class ZSPJetCorrector : public JetCorrector {
 public:
  ZSPJetCorrector (const edm::ParameterSet& fParameters);
  virtual ~ZSPJetCorrector ();
  /// apply correction using Event information 
  virtual double correction( const reco::Jet&, const edm::Event&, const edm::EventSetup& ) const;
  /// apply correction using Jet information only
  virtual double correction (const LorentzVector& fJet) const;
  /// apply correction using Jet information only
  virtual double correction (const reco::Jet& fJet) const;

  /// if correction needs event information
  virtual bool eventRequired () const {return true;}

  /// Set the number of pileups
  virtual int setPU() const {return fixedPU;}

 private:
  std::vector<SimpleZSPJetCorrector*>   mSimpleCorrector;
  std::vector<SimpleL1OffsetCorrector*> mSimpleCorrectorOffset;
  std::vector<std::string>              theFilesL1Offset;
  std::vector<std::string>              theFilesZSP;
  int                                   iPU;
  int                                   fixedPU;
};

#endif
