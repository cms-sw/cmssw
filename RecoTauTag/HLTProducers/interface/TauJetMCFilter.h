
#ifndef TauJetMCFilter_H
#define TauJetMCFilter_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/Handle.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

class TauJetMCFilter: public edm::EDFilter {
 public:
  explicit TauJetMCFilter(const edm::ParameterSet&);
  ~TauJetMCFilter();
  virtual bool filter(edm::Event&, const edm::EventSetup&);


 private:
  edm::InputTag genParticles;
  double mEtaMin, mEtaMax, mEtTau;
  typedef std::vector< HepMC::GenParticle * > GenPartVect;
  typedef std::vector< HepMC::GenParticle * >::const_iterator GenPartVectIt;
  HepMC::GenParticle * findParticle(const GenPartVect genPartVect, const int requested_id) ;

};
#endif
