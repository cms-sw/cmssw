//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: ZSPJPTJetCorrector.h,v 1.1 2012/10/18 08:46:42 eulisse Exp $
//
// MC Jet Corrector
//
#ifndef ZSPJPTJetCorrector_h
#define ZSPJPTJetCorrector_h

#include "SimpleZSPJPTJetCorrector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/JetReco/interface/Jet.h"

/// classes declaration
namespace edm {
  class ParameterSet;
}

class SimpleZSPJPTJetCorrector;

class ZSPJPTJetCorrector {
 public:
  ZSPJPTJetCorrector (const edm::ParameterSet& fParameters);
  virtual ~ZSPJPTJetCorrector ();
  /// apply correction using Event information 
  virtual double correction( const reco::Jet&, const edm::Event&, const edm::EventSetup& ) const;
  /// Set the number of pileups
  virtual int setPU() const {return fixedPU;}

 private:
  std::vector<SimpleZSPJPTJetCorrector*>   mSimpleCorrector;
  std::vector<SimpleZSPJPTJetCorrector*>   mSimpleCorrectorOffset;
  std::vector<std::string>              theFilesL1Offset;
  std::vector<std::string>              theFilesZSP;
  int                                   iPU;
  int                                   fixedPU;
};

#endif
